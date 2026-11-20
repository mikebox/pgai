"""Match what the applicant calls their vehicle to a known model name from the training data.

The credit model looks vehicles up by their exact training-data name (to get make / product / variant),
so "Jupiter BS6 Basic" must become "JUPITER BSVI BASIC" or the model treats it as an unknown vehicle.
Pure code, no LLM: fast, predictable, and identical in normal and most-secure mode.

How a name is scored:
  1. Normalise both sides: upper-case, split on spaces/punctuation, map synonyms (BS6 -> BSVI,
     BASE -> BASIC, SPECIAL -> SPL, ...), split glued tokens ("RTR160" -> RTR 160), and drop tokens
     that don't tell models apart (TVS, BSVI, "model", "bike", ...).
  2. The model family (JUPITER, APACHE, NTORQ, ...) must match, allowing small typos ("jupitor").
  3. Score = query words found in the candidate − a small penalty per extra candidate word the applicant
     didn't mention − a penalty per query word the candidate lacks. Ties go to the most common model.
So "jupiter basic" -> JUPITER BSVI BASIC, and plain "jupiter" -> the most common plain Jupiter.
The applicant sees the matched name on the review card and can change it.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

SYNONYMS = {
    "BS6": "BSVI", "BS-6": "BSVI", "BSVI": "BSVI", "BS-VI": "BSVI",
    "BASE": "BASIC", "STANDARD": "BASIC", "STD": "BASIC",
    "SPECIAL": "SPL", "SPECIALEDITION": "SPL", "EDITION": "EDN", "ED": "EDN",
    "DRUMS": "DRUM", "DISK": "DISC", "DISCS": "DISC",
    "ELECTRIC": "EV", "SCOOTER": "", "ZX": "ZX",
    "1CHANNEL": "1CH", "2CHANNEL": "2CH",
    "2VALVE": "2V", "4VALVE": "4V",
}
# Tokens that are on (nearly) every model or carry no meaning for telling models apart.
NOISE = {"TVS", "BSVI", "MODEL", "BIKE", "THE", "NEW", "VARIANT", "VERSION", "WALA", "WALI", "", "-", "U"}
FAMILIES = ["JUPITER", "APACHE", "NTORQ", "RAIDER", "RADEON", "XL", "SPORT", "SCOOTY", "IQUBE", "RONIN", "STARCITY"]
FAMILY_ALIASES = {"XL100": "XL", "STAR": "STARCITY", "PEP": "SCOOTY", "ZEST": "SCOOTY", "IQ": "IQUBE"}



def _tokens(text: str) -> list[str]:
    s = str(text or "").upper()
    s = re.sub(r"BS[\s-]*(VI|6)\b", "BSVI", s)
    s = re.sub(r"(\d)\s*(V|CH)\b", r"\1\2", s)                     # "2 V" -> 2V, "2 CH" -> 2CH
    s = re.sub(r"\b(\d)\s*(VALVE|CHANNEL)S?\b", r"\1\2", s)          # "4 valve" -> 4VALVE (-> 4V below)
    s = re.sub(r"\bSPECIAL\s+EDITION\b", "SPL EDN", s)
    s = re.sub(r"\bHEAVY\s*DUTY\b", "HD", s)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    out = []
    for t in s.split():
        # split glued letters+numbers like RTR160 / XL100 / U566 -> keep model codes like 2V, 4V, 1CH
        parts = [t] if re.fullmatch(r"\d(V|CH|P)|[A-Z]+\d{0,2}[A-Z]*", t) else re.findall(r"[A-Z]+|\d+", t)
        for p in parts:
            p = FAMILY_ALIASES.get(p, p)
            p = SYNONYMS.get(p, p)
            if p and p not in NOISE:
                out.append(p)
    return out


def _close(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.isdigit() or b.isdigit() or min(len(a), len(b)) < 4:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.8               # small typos: JUPITOR, APPACHE


def _family(tokens: list[str]) -> str | None:
    for t in tokens:
        for f in FAMILIES:
            if _close(t, f):
                return f
    return None


@lru_cache(maxsize=4)
def _index(names: tuple[str, ...]) -> list[tuple[str, list[str], str | None, int]]:
    out = []
    for rank, n in enumerate(names):
        toks = _tokens(n)
        out.append((n, toks, _family(toks), rank))
    return out


def match(query: str, names: list[str] | tuple[str, ...]) -> str | None:
    """Closest known model name for `query`, or None if nothing in the same family fits.
    `names` should be ordered most-common first (that order breaks ties)."""
    names = tuple(names)
    if not names or not str(query or "").strip():
        return None
    q = _tokens(query)
    exact = {re.sub(r"\s+", " ", n.upper().strip()): n for n in names}
    if (k := re.sub(r"\s+", " ", str(query).upper().strip())) in exact:
        return exact[k]
    fam = _family(q)
    if fam is None:
        return None
    best, best_key = None, None
    for name, toks, cfam, rank in _index(names):
        if cfam != fam:
            continue
        rest = [t for t in toks if t != fam]
        used: set[int] = set()
        hits = 0.0
        missing = 0
        for t in q:
            if _close(t, fam):
                hits += 1.0
                continue
            j = next((i for i, c in enumerate(rest) if i not in used and _close(t, c)), None)
            if j is None:
                missing += 1
            else:
                used.add(j)
                hits += 1.0
        extra = len(rest) - len(used)
        score = hits - 0.6 * missing - 0.25 * extra
        key = (score, -rank)
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best      # same family is enough: the applicant sees the match and can change it
