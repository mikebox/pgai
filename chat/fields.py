"""Registry of applicant fields the chatbot can collect, and how they're grouped.

Each field has a description (fed to the LLM), form metadata (question text,
input kind, options) and a normaliser/validator (pure Python – the LLM only
extracts from free text, code validates). Some fields are *derived* from
others (e.g. LTV from loan amount and vehicle price).

Questions are asked in themed GROUPS, one form per group.

To support a new model field: add a FieldSpec here (or a DerivedField), list
it in the model's `required_fields`, and optionally put it in a group (fields
not in any group are asked in a final "A few more details" form).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal

from .config import PROJECT_ROOT

# Validator: raw value -> (clean value, error message or None)
Validator = Callable[[Any], tuple[Any, str | None]]
Options = dict[str, tuple[str, str]]          # code -> (label, hint)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str            # short name used in summaries / errors
    description: str      # how to interpret answers (shown to the LLM)
    validate: Validator
    question: str = ""    # question shown on the form
    kind: Literal["number", "money", "text", "choice", "email", "tel"] = "text"
    options: Options | None = None
    placeholder: str = ""
    suggestions: Callable[[], list[str]] | None = None   # autocomplete values for text fields
    allow_other: bool = False   # choice fields: offer an "Other…" free-text option (mapped by the LLM)


@dataclass(frozen=True)
class DerivedField:
    name: str
    inputs: tuple[str, ...]
    compute: Callable[..., Any]


# ---- validators ----------------------------------------------------------------
def _number(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).lower().replace(",", "").replace("₹", "").replace("rs", "").strip()
    m = re.fullmatch(r"([\d.]+)\s*(k|thousand|l|lakh|lakhs|lac)?", s)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    mult = {"k": 1e3, "thousand": 1e3, "l": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5}.get(m.group(2) or "", 1)
    return n * mult


def _age(v):
    n = _number(v)
    if n is None:
        return None, "Age should be a number of years."
    n = int(n)
    if not 18 <= n <= 60:
        return None, f"Age {n} is outside our lending policy (18 to 60 years)."
    return n, None


def _pincode(v):
    s = re.sub(r"\s", "", str(v or ""))
    if not re.fullmatch(r"[1-9]\d{5}", s):
        return None, "Pincode should be a 6-digit Indian postal code."
    return s, None


def _money(label: str, lo: float, hi: float):
    def check(v):
        n = _number(v)
        if n is None:
            return None, f"{label} should be an amount in rupees."
        if not lo <= n <= hi:
            return None, f"{label} of ₹{n:,.0f} looks unusual (expected ₹{lo:,.0f}–₹{hi:,.0f}). Please check the amount."
        return round(n, 2), None
    return check


def _choice(options: Options):
    def check(v):
        s = str(v or "").strip()
        for code, (label, _) in options.items():
            if s.upper() in (code.upper(), label.upper()):
                return code, None
        return None, "Please pick one of the options."
    return check


def _text(v):
    s = str(v or "").strip()
    return (s, None) if len(s) >= 2 else (None, "Please give the vehicle make and model.")


def _name(v):
    s = re.sub(r"\s+", " ", str(v or "")).strip()
    if not 2 <= len(s) <= 80 or not re.fullmatch(r"[^\W\d_]+(?:[ .'-]+[^\W\d_]+)*\.?", s):
        return None, "Please enter your full name (letters only)."
    return s, None


def _email(v):
    s = str(v or "").strip().lower()
    if not re.fullmatch(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}", s):
        return None, "Please enter a valid email address, e.g. name@example.com."
    return s, None


def _phone(v):
    s = re.sub(r"[\s()-]", "", str(v or ""))
    s = re.sub(r"^(?:\+?91|0)(?=\d{10}$)", "", s)
    if not re.fullmatch(r"[6-9]\d{9}", s):
        return None, "Please enter a 10-digit Indian mobile number."
    return s, None


def _codes(options: Options) -> str:
    return ", ".join(f"{k}={label} ({hint})" for k, (label, hint) in options.items())


EMPLOYMENT_TYPES: Options = {
    "SAL": ("Salaried", "Fixed monthly salary from an employer"),
    "SEP": ("Self-employed", "Own business, shop or profession"),
    "AGR": ("Agriculture", "Farming or allied work"),
    "PEN": ("Pensioner", "Retired, receiving a pension"),
    "NREGI": ("Irregular income", "Daily wage, gig or seasonal work"),
    "NPP": ("Probation / temporary", "Non-permanent or probationary job"),
    "STU": ("Student", "Currently studying"),
    "NONEARNMEM": ("Not earning", "Homemaker or dependent family member"),
}
PAST_LOANS: Options = {
    "NO_PAST_LOANS": ("No", "No loan or EMI running right now"),
    "PAST_LOANS_ACTIVE": ("Yes", "Currently repaying a loan or EMI"),
}
QUALIFICATIONS: Options = {
    "SSC": ("10th", "Secondary school"), "HSC": ("12th", "Higher secondary"),
    "UG": ("Undergraduate", "Diploma / pursuing degree"), "GRAD": ("Graduate", "Bachelor's degree"),
    "PG": ("Postgraduate", "Master's or above"), "OTHERS": ("Other", ""),
}
RESIDENT_TYPES: Options = {"O": ("Owned", "Own or family-owned home"), "R": ("Rented", ""), "L": ("Leased", "")}
GENDERS: Options = {"Male": ("Male", ""), "Female": ("Female", "")}


@lru_cache(maxsize=1)
def vehicle_models() -> list[str]:
    """Known vehicle model names from the training data (for autocomplete)."""
    try:
        import pandas as pd
        for name in ("training_data_clean.csv", "training_data.csv"):
            p = PROJECT_ROOT / "data" / name
            if p.exists():
                s = pd.read_csv(p, usecols=["Model_Description"])["Model_Description"].dropna()
                return s.value_counts().index.tolist()
    except Exception:
        pass
    return []


@lru_cache(maxsize=1)
def vehicle_prices() -> dict[str, float]:
    """Typical on-road price per vehicle model, from the training data (price = loan / LTV).

    Median over that model's applications, rounded to the nearest ₹500. Used to prefill
    the price question; the applicant can always change it.
    """
    try:
        import pandas as pd
        for name in ("training_data_clean.csv", "training_data.csv"):
            p = PROJECT_ROOT / "data" / name
            if p.exists():
                d = pd.read_csv(p, usecols=["Model_Description", "Loan_Amount", "LTV"]).dropna()
                d = d[d["LTV"] > 0]
                price = (d["Loan_Amount"] / d["LTV"] * 100).groupby(d["Model_Description"]).median()
                return {m: float(round(v / 500) * 500) for m, v in price.items()}
    except Exception:
        pass
    return {}


FIELDS: dict[str, FieldSpec] = {f.name: f for f in [
    # Contact details: always collected (step 1), never sent to the credit model, and never sent to
    # the LLM in clear text – see chat/pii.py. The LLM only ever sees placeholders like [EMAIL_1].
    FieldSpec("Name", "name", "Applicant's full name. Only ever appears as a placeholder like [NAME_1]; "
              "return that placeholder exactly as written.", _name,
              question="What's your full name?", kind="text", placeholder="As on your PAN / Aadhaar"),
    FieldSpec("Email", "email", "Applicant's email. Only ever appears as a placeholder like [EMAIL_1]; "
              "return that placeholder exactly as written.", _email,
              question="Your email address", kind="email", placeholder="name@example.com"),
    FieldSpec("Phone", "mobile number", "Applicant's mobile number. Only ever appears as a placeholder like "
              "[PHONE_1]; return that placeholder exactly as written.", _phone,
              question="Your mobile number", kind="tel", placeholder="10-digit mobile number"),
    FieldSpec("Age", "age", "Applicant's age in whole years.", _age,
              question="How old are you?", kind="number", placeholder="e.g. 28"),
    FieldSpec("Pincode", "pincode", "6-digit postal code of where the applicant lives.", _pincode,
              question="What's your area pincode?", kind="number", placeholder="6-digit pincode"),
    FieldSpec("Employment_Type", "employment type", "Occupation category, one of: " + _codes(EMPLOYMENT_TYPES),
              _choice(EMPLOYMENT_TYPES), question="What do you do for a living?", kind="choice", options=EMPLOYMENT_TYPES,
              allow_other=True),
    FieldSpec("Net_salary", "monthly income",
              "Self-reported net MONTHLY income in INR (convert annual to monthly only if the user clearly says annual).",
              _money("Monthly income", 1_000, 2_000_000),
              question="What's your monthly take-home income?", kind="money", placeholder="e.g. 35,000"),
    FieldSpec("PAST_LOANS_ACTIVE", "existing loans",
              "Whether the applicant currently has any active loan: " + _codes(PAST_LOANS), _choice(PAST_LOANS),
              question="Are you repaying any loan or EMI right now?", kind="choice", options=PAST_LOANS),
    FieldSpec("Model_Description", "vehicle model",
              "The two-wheeler being financed: make and model as the user names it, e.g. 'TVS Jupiter 125', 'Apache RTR 160'.",
              _text, question="Which vehicle are you buying?", placeholder="Start typing, e.g. Jupiter, Raider, Apache",
              suggestions=vehicle_models),
    FieldSpec("Vehicle_Price", "vehicle price", "On-road price of the vehicle in INR (used to compute LTV).",
              _money("Vehicle price", 10_000, 1_500_000),
              question="What's its on-road price?", kind="money", placeholder="e.g. 1,10,000"),
    FieldSpec("Loan_Amount", "loan amount", "Loan amount requested in INR.", _money("Loan amount", 5_000, 1_000_000),
              question="How much loan do you need?", kind="money", placeholder="e.g. 85,000"),
    # Available if a future model needs them:
    FieldSpec("Qualifications", "education", "Highest qualification: " + _codes(QUALIFICATIONS), _choice(QUALIFICATIONS),
              question="What's your highest qualification?", kind="choice", options=QUALIFICATIONS,
              allow_other=True),
    FieldSpec("Resident_Type", "residence type", "Home is " + _codes(RESIDENT_TYPES), _choice(RESIDENT_TYPES),
              question="Is your home owned or rented?", kind="choice", options=RESIDENT_TYPES),
    FieldSpec("Gender", "gender", "Male or Female.", _choice(GENDERS),
              question="Gender", kind="choice", options=GENDERS),
]}


# (id, title shown on the form header, fields in order)
# Asked of every applicant regardless of the model (step 1). PII: kept out of the model input and the LLM.
CONTACT_FIELDS: list[str] = ["Name", "Email", "Phone"]

GROUPS: list[tuple[str, str, list[str]]] = [
    ("contact", "Your contact details", CONTACT_FIELDS),
    ("about", "About you", ["Age", "Pincode", "Employment_Type", "Qualifications", "Resident_Type", "Gender"]),
    ("finances", "Your finances", ["Net_salary", "PAST_LOANS_ACTIVE"]),
    ("vehicle", "Your vehicle", ["Model_Description", "Vehicle_Price", "Loan_Amount"]),
]


def _ltv(loan: float, price: float) -> float:
    return round(100.0 * loan / price, 2)


DERIVED: dict[str, DerivedField] = {
    "LTV": DerivedField("LTV", ("Loan_Amount", "Vehicle_Price"), _ltv),
}


def fields_to_collect(model_fields: list[str]) -> list[str]:
    """Expand the model's required fields into fields we must ask the user for (in group order)."""
    need: list[str] = list(CONTACT_FIELDS)
    for f in model_fields:
        for n in (DERIVED[f].inputs if f in DERIVED else (f,)):
            if n not in FIELDS:
                raise KeyError(f"Model needs field {n!r} but chat/fields.py has no FieldSpec for it.")
            if n not in need:
                need.append(n)
    return [f for g in groups_for(need) for f in g[2]]


def groups_for(collect: list[str]) -> list[tuple[str, str, list[str]]]:
    """GROUPS filtered to the fields being collected; ungrouped fields go into a final group."""
    out, seen = [], set()
    for gid, title, names in GROUPS:
        fs = [n for n in names if n in collect]
        seen.update(fs)
        if fs:
            out.append((gid, title, fs))
    rest = [n for n in collect if n not in seen]
    if rest:
        out.append(("more", "A few more details", rest))
    # A step with a single question isn't worth an extra turn: fold it into the step before.
    merged: list[tuple[str, str, list[str]]] = []
    for gid, title, fs in out:
        if len(fs) == 1 and merged:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + fs)
        else:
            merged.append((gid, title, fs))
    return merged


def field_ui(name: str, value=None, error: str | None = None) -> dict:
    """JSON-able description of one form field for the web UI."""
    s = FIELDS[name]
    d = {"name": name, "label": s.label, "question": s.question or s.label, "kind": s.kind,
         "placeholder": s.placeholder, "value": value, "error": error}
    if s.options:
        d["options"] = [{"value": k, "label": lab, "hint": hint} for k, (lab, hint) in s.options.items()]
        d["allow_other"] = s.allow_other
    if s.suggestions:
        d["suggestions"] = s.suggestions()
    if name == "Model_Description":
        d["prices"] = vehicle_prices()          # model -> typical price, to prefill Vehicle_Price
        d["fills"] = "Vehicle_Price"
    if name == "Loan_Amount":
        # Slider as a % of the vehicle price (that % is the LTV). Loan can't exceed the price.
        d["slider"] = {"of": "Vehicle_Price", "min_pct": 10, "max_pct": 100, "default_pct": 80}
    return d


def inr(n: float) -> str:
    """Indian digit grouping: 150000 -> ₹1,50,000."""
    s = str(int(round(n)))
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail = head[-2:] + "," + tail
        head = head[:-2]
    return "₹" + (head + "," + tail if head else tail)


def display(name: str, v) -> str:
    """Human-friendly value for summaries."""
    s = FIELDS[name]
    if v is None:
        return ""
    if s.kind == "money":
        return inr(v)
    if s.options and v in s.options:
        return s.options[v][0]
    return str(v)


def cross_validate(app: dict) -> dict[str, str]:
    """Checks that involve more than one field."""
    errs = {}
    if "Loan_Amount" in app and "Vehicle_Price" in app and app["Loan_Amount"] > app["Vehicle_Price"]:
        errs["Loan_Amount"] = "The loan amount can't be more than the vehicle's price."
    return errs


def build_model_input(app: dict, model_fields: list[str]) -> dict:
    """Collected answers -> the exact dict of fields the ML model consumes."""
    row = {}
    for f in model_fields:
        if f in DERIVED:
            d = DERIVED[f]
            row[f] = d.compute(*(app[i] for i in d.inputs))
        else:
            row[f] = app.get(f)
    return row
