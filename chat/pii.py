"""PII redaction at the LLM boundary.

Nothing personally identifying is sent to the LLM in clear text. Before every
call, the prompt is passed through a `Redactor` that swaps PII for stable
placeholders ([NAME_1], [EMAIL_1], [PHONE_1], [PAN_1], [AADHAAR_1], [ACCOUNT_1]).
The mapping stays in this process; the LLM's output is "restored" locally, so

  * a reply like "Thanks, [NAME_1]!" is shown to the applicant as "Thanks, Ravi!"
  * an extraction like Email="[EMAIL_1]" becomes the real address before validation.

What gets redacted:
  * the applicant's own name / email / phone once collected (exact-value match,
    including each part of the name, so "Ravi" alone is caught after "Ravi Kumar")
  * any email address, Indian mobile number, PAN, Aadhaar or long account/card number
  * names introduced in free text ("my name is …", "Name: …", "call me …")

Age, pincode, income and vehicle details are NOT redacted: the LLM needs them to
read the applicant's answers, and none identifies a person on its own.

`PIISafeLLM` wraps any `chat.llm.LLM` and applies this to every call.
"""
from __future__ import annotations

import re
from typing import Iterable

from .fields import CONTACT_FIELDS

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I)),
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}\b")),
    ("ACCOUNT", re.compile(r"\b\d{11,18}\b")),
]
# "my name is Ravi Kumar", "Name: ravi", "call me Ravi", "this is Ravi" – group 1 is the name.
# After the intro phrase the first word may be lower-case; further words must be Capitalised.
_NOT_NAMES = {"a", "an", "the", "and", "i", "my", "me", "not", "for", "about", "it", "is", "from", "here",
              "to", "regarding", "looking", "applying", "interested", "your", "our", "new", "first", "na",
              "on", "at", "back", "later", "now", "today", "tomorrow", "please", "sir", "madam",
              # after "I'm …": states, not names
              "salaried", "self", "self-employed", "employed", "unemployed", "retired", "student", "working",
              "married", "single", "fine", "good", "ok", "okay", "also", "just", "currently", "planning",
              "buying", "trying", "sure", "ready", "done", "happy", "sorry", "using", "living", "staying",
              "earning", "getting", "paying", "having", "going", "doing", "with", "in", "into", "very", "still",
              "farmer", "pensioner", "shopkeeper", "tvs", "jupiter", "apache", "ntorq", "raider", "indian"}
_NAME_INTRO = re.compile(
    r"(?:(?i:\bmy name is|\bmy name's|\bname\s*[:=-]|\bcall me|\bi am called|\bi'm called)\s+"
    r"([^\W\d_][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3})"
    r"|(?i:\bthis is|\bi am|\bi'm|\bi’m|\bim)\s+([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3}))")
_TOKEN = re.compile(r"\[(NAME|EMAIL|PHONE|PAN|AADHAAR|ACCOUNT)_\d+\]")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


class Redactor:
    """Reversible pseudonymisation for one LLM call (or a few calls on the same session)."""

    def __init__(self, known: dict | None = None):
        self._to_token: dict[tuple[str, str], str] = {}    # (kind, normalised value) -> token
        self._to_value: dict[str, str] = {}                 # token -> original value
        self._counts: dict[str, int] = {}
        self._known: list[tuple[str, str]] = []             # (kind, value) of the applicant's own details
        known = known or {}
        for field, kind in (("Name", "NAME"), ("Email", "EMAIL"), ("Phone", "PHONE")):
            v = known.get(field)
            if v:
                self._token(kind, str(v))                   # applicant's own details are always *_1
                self._known.append((kind, str(v)))

    # ---- mapping -------------------------------------------------------------------
    @staticmethod
    def _norm(kind: str, value: str) -> str:
        if kind in ("PHONE", "AADHAAR", "ACCOUNT"):
            d = _digits(value)
            return d[-10:] if kind == "PHONE" else d
        return value.strip().lower()

    def _token(self, kind: str, value: str) -> str:
        key = (kind, self._norm(kind, value))
        if key not in self._to_token:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            tok = f"[{kind}_{self._counts[kind]}]"
            self._to_token[key] = tok
            self._to_value[tok] = value.strip()
        return self._to_token[key]

    # ---- redact / restore ------------------------------------------------------------
    def redact(self, text: str) -> str:
        if not text:
            return text
        known = dict((k, v) for k, v in self._known)
        # 1. The applicant's own email / phone, exact match (phone: digits with any separators).
        if "EMAIL" in known:
            text = re.sub(re.escape(known["EMAIL"]), self._token("EMAIL", known["EMAIL"]), text, flags=re.I)
        if "PHONE" in known:
            d = self._norm("PHONE", known["PHONE"])
            pat = r"(?<!\d)(?:\+?91[\s-]?|0)?" + r"[\s-]?".join(d) + r"(?!\d)"
            text = re.sub(pat, self._token("PHONE", known["PHONE"]), text)
        # 2. Structured identifiers anywhere in the text (before names, so "ravi@…" goes as one email).
        for kind, pat in _PATTERNS:
            text = pat.sub(lambda m, k=kind: self._token(k, m.group(0)), text)
        # 3. The applicant's own name, in full or any part of it ("Ravi" after "Ravi Kumar").
        if "NAME" in known:
            tok = self._token("NAME", known["NAME"])
            text = re.sub(re.escape(known["NAME"]), tok, text, flags=re.I)
            for part in self._name_parts(known["NAME"]):
                text = re.sub(rf"(?<![\w\[]){re.escape(part)}(?![\w\]])", tok, text, flags=re.I)
        # 4. Other names introduced in free text.
        def name(m: re.Match) -> str:
            g = 1 if m.group(1) else 2
            if m.group(g).split()[0].lower() in _NOT_NAMES:
                return m.group(0)
            return m.group(0)[: m.start(g) - m.start(0)] + self._token("NAME", m.group(g))
        return _NAME_INTRO.sub(name, text)

    def restore(self, text):
        if not isinstance(text, str):
            return text
        return _TOKEN.sub(lambda m: self._to_value.get(m.group(0), m.group(0)), text)

    @staticmethod
    def _name_parts(name: str) -> Iterable[str]:
        return [p for p in re.split(r"[\s.]+", name) if len(p) >= 3]

    def token_for(self, field: str) -> str | None:
        """Placeholder of the applicant's own Name / Email / Phone, if known."""
        kind = {"Name": "NAME", "Email": "EMAIL", "Phone": "PHONE"}[field]
        return next((self._token(k, v) for k, v in self._known if k == kind), None)

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._to_value)


def redact_text(text: str, known: dict | None = None) -> str:
    """One-way redaction (no restore), e.g. for logs and traces. `known`: the applicant's answers, if any."""
    return Redactor({k: v for k, v in (known or {}).items() if k in CONTACT_FIELDS}).redact(text)


class PIISafeLLM:
    """Wraps an LLM so prompts are redacted before the call and outputs restored after it.

    `known` is the applicant's collected answers; only the contact fields are used.
    """

    def __init__(self, llm, known: dict | None = None):
        self.llm = llm
        self.redactor = Redactor({k: v for k, v in (known or {}).items() if k in CONTACT_FIELDS})

    def extract(self, transcript: str, collected: dict, wanted: list[str]):
        r = self.redactor
        tokens = {k: r.token_for(k) for k in CONTACT_FIELDS}
        safe_collected = {k: (tokens[k] or "[REDACTED]") if k in CONTACT_FIELDS else v for k, v in collected.items()}
        redacted = r.redact(transcript)
        ex = self.llm.extract(redacted, safe_collected, wanted)
        restored = {k: r.restore(v) for k, v in ex.fields.model_dump().items()}
        # A contact value that isn't one of our placeholders was never shown to the LLM, so it's invented.
        for k in CONTACT_FIELDS:
            v = ex.fields.model_dump().get(k)
            if v and not _TOKEN.fullmatch(str(v).strip()):
                restored[k] = None
        # The model sometimes leaves a contact field empty even though the applicant just gave it
        # ("my name is Jay, email jay@test.com …"). The redactor already knows what each placeholder stands for,
        # so if the latest applicant message holds exactly one placeholder of that kind, use it.
        latest = redacted.rsplit("Applicant:", 1)[-1]
        for k, kind in (("Name", "NAME"), ("Email", "EMAIL"), ("Phone", "PHONE")):
            if k in CONTACT_FIELDS and k in wanted and not restored.get(k):
                toks = set(re.findall(rf"\[{kind}_\d+\]", latest))
                if len(toks) == 1:
                    restored[k] = r.restore(toks.pop())
        return ex.model_copy(update={"fields": type(ex.fields)(**restored),
                                     "ambiguities": [r.restore(a) for a in ex.ambiguities]})

    def reply(self, transcript: str, instructions: str) -> str:
        r = self.redactor
        return r.restore(self.llm.reply(r.redact(transcript), r.redact(instructions)))
