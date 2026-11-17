"""Append-only JSONL audit log with PII masking.

Logs every user input, extracted answer, model feature values, score and
decision. Free text is masked (phone numbers, emails, long digit runs, ID-like
tokens); pincode is truncated to its first 3 digits.
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


def mask_text(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def mask_fields(d: dict) -> dict:
    """Mask pincodes (any key spelled 'pincode', at any depth) to their first 3 digits."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = mask_fields(v)
        elif k.lower() == "pincode" and v:
            out[k] = str(v)[:3] + "***"
        else:
            out[k] = v
    return out


def log_event(session_id: str, event: str, **data) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "session_id": session_id, "event": event, **data}
    config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(config.AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
