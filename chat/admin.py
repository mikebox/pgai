"""Admin view data: rebuilds every application from the JSONL audit log.

The audit log is the only store that survives a server restart, so it is the
source of truth here. Values are shown as logged, i.e. already PII-masked
(pincode = first 3 digits, free text masked).

One application = one session_id. Status is derived from its events:
    not_started  session opened, no answers yet
    in_progress  some answers given, activity within ABANDON_AFTER
    review       every question answered, waiting for the applicant to submit
    abandoned    not decided and idle for longer than ABANDON_AFTER
    approved / declined   the ML model's decision
"""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timedelta, timezone

from . import config
from .fields import FIELDS, display

ABANDON_AFTER = timedelta(minutes=30)
STATUSES = ["approved", "declined", "review", "in_progress", "not_started", "abandoned"]

_cache: dict = {"key": None, "apps": []}
_lock = threading.Lock()


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _clean(v):
    """NaN -> None (at any depth) so the payload is valid JSON."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return None if isinstance(v, float) and math.isnan(v) else v


def _read_events() -> list[dict]:
    path = config.AUDIT_LOG
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))      # json accepts the NaN the logger writes
            except json.JSONDecodeError:
                continue
    return out


def _build(events: list[dict], collect: list[str]) -> list[dict]:
    by_sid: dict[str, dict] = {}
    for e in events:
        sid = e.get("session_id")
        if not sid or "ts" not in e:
            continue
        a = by_sid.setdefault(sid, {"id": sid, "events": [], "answers": {}, "errors": {}, "prediction": None,
                                    "model": None, "model_version": None, "inputs": 0, "text_messages": 0,
                                    "timing": None, "last_submit": None})
        a["events"].append(e)
        ev = e.get("event")
        if ev == "session_start":
            a["model"], a["model_version"] = e.get("model"), e.get("model_version")
        elif ev == "decision_timing":
            a["timing"] = e
            continue                              # bookkeeping, not applicant activity
        if ev == "user_submit":
            a["last_submit"] = e["ts"]
        if ev in ("user_form", "user_message", "user_submit"):
            a["inputs"] += 1
            a["text_messages"] += ev == "user_message"
        elif ev == "extraction":
            a["answers"] = dict(e.get("application") or {})
            a["errors"] = dict(e.get("errors") or {})
        elif ev == "prediction":
            a["prediction"] = e
            a["model"], a["model_version"] = e.get("model") or a["model"], e.get("model_version") or a["model_version"]

    now = datetime.now(timezone.utc)
    apps = []
    for a in by_sid.values():
        evs = [e for e in a["events"] if e.get("event") != "decision_timing"]
        started, last = _ts(evs[0]["ts"]), _ts(evs[-1]["ts"])
        answers, pred = a["answers"], a["prediction"]
        ltv = None
        if pred:
            inputs = (pred.get("features") or {}).get("inputs") or pred.get("features") or {}
            for k, v in inputs.items():
                if k in FIELDS and k not in answers:
                    answers[k] = v
            ltv = inputs.get("LTV")
            status = "approved" if pred.get("decision") == "APPROVED" else "declined"
            finished = _ts(pred["ts"])
        else:
            finished = None
            if not answers:
                status = "not_started"
            elif all(f in answers for f in collect) and not a["errors"]:
                status = "review"
            else:
                status = "in_progress"
            if now - last > ABANDON_AFTER:
                status = "abandoned"
        if ltv is None and answers.get("Loan_Amount") and answers.get("Vehicle_Price"):
            ltv = round(100.0 * answers["Loan_Amount"] / answers["Vehicle_Price"], 2)

        # Decision time (final input -> decision). Server-measured when logged; older sessions fall back
        # to the gap between the submit and prediction log lines.
        decision_ms = None
        if a["timing"]:
            decision_ms = a["timing"].get("decision_ms")
        elif pred and a["last_submit"]:
            decision_ms = round((_ts(pred["ts"]) - _ts(a["last_submit"])).total_seconds() * 1000, 1)
        end = finished or last
        apps.append({
            "id": a["id"],
            "status": status,
            "started_at": started.isoformat(),
            "last_activity_at": last.isoformat(),
            "decided_at": finished.isoformat() if finished else None,
            "duration_s": round((end - started).total_seconds(), 1),
            "decision_ms": decision_ms,
            "decision_measured": bool(a["timing"]),
            "within_sla": None if decision_ms is None else decision_ms <= config.DECISION_SLA_S * 1000,
            "answered": sum(f in answers for f in collect),
            "total_questions": len(collect),
            "inputs": a["inputs"],
            "text_messages": a["text_messages"],
            "answers": {k: _clean(v) for k, v in answers.items()},
            "answers_display": {k: display(k, v) for k, v in answers.items() if k in FIELDS and v is not None},
            "open_errors": a["errors"],
            "ltv": _clean(ltv),
            "decision": pred.get("decision") if pred else None,
            "probability_decline": _clean(pred.get("probability_decline")) if pred else None,
            "threshold": pred.get("threshold") if pred else None,
            "model_latency_ms": pred.get("latency_ms") if pred else None,
            "is_placeholder": pred.get("is_placeholder") if pred else a["model"] == "placeholder",
            "model": a["model"],
            "model_version": a["model_version"],
        })
    return apps


def all_applications(collect: list[str]) -> tuple[list[dict], dict[str, list[dict]]]:
    """(applications, events by session). Re-parses the log only when it changes."""
    path = config.AUDIT_LOG
    st = path.stat() if path.exists() else None
    # Status depends on the clock (abandoned), so bucket the key by minute too.
    key = (str(path), st.st_mtime_ns if st else 0, st.st_size if st else 0, tuple(collect),
           datetime.now(timezone.utc).strftime("%Y%m%d%H%M"))
    with _lock:
        if _cache["key"] != key:
            events = _read_events()
            by_sid: dict[str, list[dict]] = {}
            for e in events:
                by_sid.setdefault(e.get("session_id"), []).append(e)
            _cache.update(key=key, apps=_build(events, collect), events=by_sid)
        return _cache["apps"], _cache["events"]


def _matches(app: dict, q: str) -> bool:
    hay = [app["id"], app["status"].replace("_", " "), app.get("model") or ""]
    hay += [str(v) for v in app["answers"].values() if v is not None]
    hay += list(app["answers_display"].values())
    return any(q in h.lower() for h in hay)


SORTS = {
    "started_desc": (lambda a: a["started_at"], True),
    "started_asc": (lambda a: a["started_at"], False),
    "duration_desc": (lambda a: a["duration_s"], True),
    "duration_asc": (lambda a: a["duration_s"], False),
    "loan_desc": (lambda a: a["answers"].get("Loan_Amount") or 0, True),
    "decision_desc": (lambda a: a["decision_ms"] if a["decision_ms"] is not None else -1, True),
    "risk_desc": (lambda a: a["probability_decline"] if a["probability_decline"] is not None else -1, True),
    "activity_desc": (lambda a: a["last_activity_at"], True),
}


def query(collect: list[str], params: dict[str, str]) -> dict:
    apps, _ = all_applications(collect)
    q = (params.get("q") or "").strip().lower()
    status = [s for s in (params.get("status") or "").split(",") if s]
    emp = params.get("employment") or ""
    loans = params.get("loans") or ""
    date_from, date_to = params.get("from") or "", params.get("to") or ""
    try:   # browser's getTimezoneOffset(): minutes to ADD to local time to get UTC (IST = -330)
        local = timezone(timedelta(minutes=-int(params.get("tz") or 0)))
    except ValueError:
        local = timezone.utc

    def keep(a: dict) -> bool:
        if status and a["status"] not in status:
            return False
        if emp and a["answers"].get("Employment_Type") != emp:
            return False
        if loans and a["answers"].get("PAST_LOANS_ACTIVE") != loans:
            return False
        day = _ts(a["started_at"]).astimezone(local).date().isoformat()
        if date_from and day < date_from:
            return False
        if date_to and day > date_to:
            return False
        return not q or _matches(a, q)

    rows = [a for a in apps if keep(a)]
    keyfn, rev = SORTS.get(params.get("sort") or "", SORTS["started_desc"])
    rows.sort(key=keyfn, reverse=rev)

    try:
        size = max(1, min(100, int(params.get("page_size") or 20)))
        page = max(1, int(params.get("page") or 1))
    except ValueError:
        size, page = 20, 1
    pages = max(1, math.ceil(len(rows) / size))
    page = min(page, pages)

    decided = [a for a in rows if a["decision"]]
    durations = sorted(a["duration_s"] for a in decided)
    summary = {
        "total": len(rows),
        "by_status": {s: sum(a["status"] == s for a in rows) for s in STATUSES},
        "approval_rate": (sum(a["decision"] == "APPROVED" for a in decided) / len(decided)) if decided else None,
        "median_completion_s": durations[len(durations) // 2] if durations else None,
        "sla_s": config.DECISION_SLA_S,
        "decision_ms_median": (lambda v: v[len(v) // 2] if v else None)(sorted(a["decision_ms"] for a in decided if a["decision_ms"] is not None)),
        "decision_ms_max": max((a["decision_ms"] for a in decided if a["decision_ms"] is not None), default=None),
        "within_sla": sum(bool(a["within_sla"]) for a in decided),
        "timed": sum(a["decision_ms"] is not None for a in decided),
    }
    return {"items": rows[(page - 1) * size: page * size],
            "page": page, "page_size": size, "pages": pages, "total": len(rows), "summary": summary,
            "employment_types": {k: s[0] for k, s in (FIELDS["Employment_Type"].options or {}).items()},
            "loan_options": {k: s[0] for k, s in (FIELDS["PAST_LOANS_ACTIVE"].options or {}).items()}}


def detail(collect: list[str], sid: str) -> dict | None:
    apps, events = all_applications(collect)
    app = next((a for a in apps if a["id"] == sid), None)
    if app is None:
        return None
    pred = next((e for e in reversed(events.get(sid, [])) if e.get("event") == "prediction"), None)
    engineered = ((pred or {}).get("features") or {}).get("engineered") or {}
    return {**app,
            "labels": {k: FIELDS[k].label for k in app["answers"] if k in FIELDS},
            "engineered": {k: _clean(v) for k, v in engineered.items()},
            "timeline": sorted(
                [_clean({k: v for k, v in e.items() if k != "session_id"})
                 for e in events.get(sid, []) if e.get("event") != "prediction"]
                + ([_clean({"ts": pred["ts"], "event": "prediction", "decision": pred.get("decision"),
                            "probability_decline": pred.get("probability_decline"),
                            "threshold": pred.get("threshold"), "latency_ms": pred.get("latency_ms")})]
                   if pred else []),
                key=lambda e: e["ts"])}


# ---- model / training report ----------------------------------------------------------
REPORTS_DIR = config.PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_META = config.PROJECT_ROOT / "models" / "loan_decision_model.json"

# file stem -> (section, title, what it shows)
FIGURES = {
    "01_target_balance": ("data", "Target balance", "Only 5.3% of historical applications were declined, so accuracy alone is misleading."),
    "02_missing_values": ("data", "Missing values", "Share of missing values per field before cleaning."),
    "03_numeric_distributions": ("data", "Numeric distributions", "Distributions of age, salary, loan amount and LTV."),
    "04_categorical_decline_rates": ("data", "Decline rate by category", "How the decline rate varies across each categorical field."),
    "05_mutual_information": ("features", "Mutual information", "How much each field tells us about the decision on its own."),
    "06_numeric_decline_by_decile": ("data", "Decline rate by decile", "Decline rate across deciles of each numeric field; LTV shows the steepest rise."),
    "07_ltv_loan_kde": ("data", "LTV and loan amount", "Density of LTV and loan amount for approved vs declined applications."),
    "08_decline_by_pincode_prefix": ("data", "Decline rate by postal circle", "Geography matters: decline rates range from 1.3% to 12.9%."),
    "09_interactions": ("data", "Interactions", "Joint effects between the strongest fields."),
    "10_numeric_correlation": ("features", "Correlation", "Correlation between numeric fields."),
    "11_forward_selection": ("features", "Forward selection", "CV ROC-AUC as questions are added; it plateaus at 7 questions."),
    "12_threshold_selection": ("threshold", "Threshold selection", "Expected cost by threshold at a 10:1 cost ratio; the minimum is at 0.09."),
    "13_confusion_matrices": ("test", "Confusion matrices", "Test-set outcomes at the chosen threshold vs the default 0.5."),
    "14_roc_pr_calibration": ("test", "ROC, PR and calibration", "Ranking quality and how well predicted probabilities match reality."),
    "15_permutation_importance": ("test", "Permutation importance", "Which inputs the final model relies on most."),
}


def _json_file(p):
    if not p.exists():
        return None
    return _clean(json.loads(p.read_text(encoding="utf-8")))


def report(collect: list[str]) -> dict:
    meta = _json_file(MODEL_META) or {}
    metrics = _json_file(REPORTS_DIR / "metrics.json") or meta.get("metrics") or {}
    meta.pop("metrics", None)
    md = REPORTS_DIR / "report.md"
    figs = [{"file": p.name, "section": FIGURES.get(p.stem, ("other",))[0],
             "title": FIGURES.get(p.stem, ("", p.stem.replace("_", " ")))[1],
             "caption": FIGURES.get(p.stem, ("", "", ""))[2]}
            for p in sorted(FIGURES_DIR.glob("*.png"))] if FIGURES_DIR.exists() else []

    apps, _ = all_applications(collect)
    decided = [a for a in apps if a["decision"] and not a["is_placeholder"]]
    probs = [a["probability_decline"] for a in decided if a["probability_decline"] is not None]
    lat = sorted(a["model_latency_ms"] for a in decided if a["model_latency_ms"] is not None)
    live = {
        "applications": len(apps),
        "decided": len(decided),
        "declined": sum(a["decision"] == "DECLINED" for a in decided),
        "decline_rate": (sum(a["decision"] == "DECLINED" for a in decided) / len(decided)) if decided else None,
        "mean_probability_decline": (sum(probs) / len(probs)) if probs else None,
        "median_latency_ms": lat[len(lat) // 2] if lat else None,
        "median_completion_s": sorted(a["duration_s"] for a in decided)[len(decided) // 2] if decided else None,
    }
    return {"model": meta, "metrics": metrics, "report_md": md.read_text(encoding="utf-8") if md.exists() else "",
            "figures": figs, "live": live}


def figure_path(name: str):
    """Only plain PNG names directly inside reports/figures."""
    if "/" in name or "\\" in name or not name.endswith(".png"):
        return None
    p = FIGURES_DIR / name
    return p if p.is_file() else None
