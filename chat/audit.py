"""Append-only JSONL audit log with PII masking.

Logs every user input, extracted answer, model feature values, score and
decision. Free text is masked (phone numbers, emails, long digit runs, ID-like
tokens); pincode is truncated to its first 3 digits; name, email and phone
fields are partially masked (initials, domain, last 4 digits).
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone

from . import config

_lock = threading.Lock()

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I), "[PAN]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b(\d{3})\d{3}\b"), r"\1***"),            # 6-digit pincodes
    (re.compile(r"\b\d{9,18}\b"), "[NUMBER]"),               # account / card-like numbers
]


def mask_text(text: str, known: dict | None = None) -> str:
    """`known`: the applicant's collected answers, so their own name is masked wherever it appears."""
    from .pii import redact_text      # names typed in free text ("my name is …"), emails, phones, IDs
    text = redact_text(text, known)
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def mask_value(key: str, v):
    """Partial mask for one PII field value (None if the key isn't a PII field)."""
    k, s = key.lower(), str(v)
    if k == "pincode":
        return s[:3] + "***"
    if k == "name":
        return " ".join(p[:1] + "***" for p in s.split())
    if k == "email":
        user, _, domain = s.partition("@")
        return user[:1] + "***@" + domain
    if k == "phone":
        return "******" + re.sub(r"\D", "", s)[-4:]
    return None


def mask_fields(d: dict) -> dict:
    """Mask PII fields (pincode, name, email, phone – any depth) and free text in the rest."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = mask_fields(v)
        elif v and (m := mask_value(k, v)) is not None:
            out[k] = m
        else:
            out[k] = v
    return out


def log_event(session_id: str, event: str, **data) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "session_id": session_id, "event": event, **data}
    config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(config.AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
