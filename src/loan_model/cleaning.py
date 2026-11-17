"""Data loading and cleaning.

Every rule here was motivated by a finding in notebooks/01_eda.ipynb.
`clean_raw` is deterministic and row-wise (no statistics learned from data), so
it is safe to apply identically to training data, the hold-out test set and
single chatbot applications. Learned transforms (imputation, encoding,
winsorisation) live in features.py inside the sklearn Pipeline.
"""
from __future__ import annotations



import numpy as np
import pandas as pd

from . import config as C

CATEGORICAL_COLS = [
    "Qualifications", "Employment_Type", "Gender", "Product_Code",
    "Resident_Type", "Final_Tier", "Make_Code", "Model_Description",
    "Model_Variant", "PAST_LOANS_ACTIVE",
]
NUMERIC_COLS = ["Age", "Loan_Amount", "LTV", "Net_salary"]

# Mis-decoded UTF-8 (en dash "–" read as cp1252) seen in Model_Description.
_MOJIBAKE = {"â€“": "-", "â€”": "-", "â€™": "'", "Ã©": "e"}


def load_raw(path=C.RAW_DATA) -> pd.DataFrame:
    """Read the CSV with everything as string so nothing is silently coerced."""
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _norm_text(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    for bad, good in _MOJIBAKE.items():
        s = s.str.replace(bad, good, regex=False)
    s = s.str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"\s*-\s*", " - ", regex=True)
    return s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})


def _norm_variant(s: pd.Series) -> pd.Series:
    # "125CC DISC" and "125 CC DISC" are the same variant; "RTR 200CC" -> "RTR 200 CC".
    return s.str.replace(r"(\d)\s*CC\b", r"\1 CC", regex=True)


def _norm_pincode(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    valid = s.str.fullmatch(r"[1-9]\d{5}").fillna(False)
    return s.where(valid, pd.NA)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning. Returns a new frame with typed columns and QA flags."""
    out = df.copy()

    # Text / categorical normalisation
    for col in CATEGORICAL_COLS:
        if col in out:
            out[col] = _norm_text(out[col])
    if "Model_Variant" in out:
        out["Model_Variant"] = _norm_variant(out["Model_Variant"])
    if "Gender" in out:
        out["Gender"] = out["Gender"].str.title()
    if "Pincode" in out:
        out["Pincode"] = _norm_pincode(out["Pincode"])

    # Numerics: blanks / junk -> NaN
    for col in NUMERIC_COLS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    # Impossible values -> NaN (imputed later, with a missing-indicator)
    if "LTV" in out:
        bad = (out["LTV"] < C.MIN_PLAUSIBLE_LTV) | (out["LTV"] > C.MAX_PLAUSIBLE_LTV)
        out.loc[bad, "LTV"] = np.nan
    if "Net_salary" in out:
        out.loc[out["Net_salary"] < C.MIN_PLAUSIBLE_MONTHLY_SALARY, "Net_salary"] = np.nan
    if "Age" in out:
        out.loc[(out["Age"] < 16) | (out["Age"] > 100), "Age"] = np.nan

    # Final_Tier is a pure function of Pincode in the data, so fill it from a lookup
    # when a pincode is known (done in features.py with the learned map).

    if C.TARGET in out:
        out[C.TARGET] = _norm_text(out[C.TARGET])
    return out


def drop_unusable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Training-time only: remove rows that carry no usable information."""
    report = {"rows_in": len(df)}
    feature_cols = [c for c in df.columns if c not in (C.ID_COL, C.TARGET)]

    no_target = df[C.TARGET].isna() | ~df[C.TARGET].isin(["APPROVED", "DECLINED"])
    report["dropped_no_target"] = int(no_target.sum())
    df = df[~no_target]

    all_empty = df[feature_cols].isna().all(axis=1)
    report["dropped_all_features_empty"] = int(all_empty.sum())
    df = df[~all_empty]

    dup = df.duplicated(subset=feature_cols + [C.TARGET], keep="first")
    report["dropped_exact_duplicates_excl_id"] = int(dup.sum())
    df = df[~dup]

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def target_to_int(y: pd.Series) -> np.ndarray:
    return (y == C.POSITIVE_LABEL).astype(int).to_numpy()


