"""Feature engineering as sklearn-compatible transformers.

Everything learned from data (lookups, caps) is learned in `fit`, so the
whole thing lives inside one Pipeline and never leaks test information.
Unseen categories at inference time map to NaN / "UNKNOWN" and never raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

UNKNOWN = "UNKNOWN"

# Derived from a vehicle model choice – the chatbot asks "which model?" only.
VEHICLE_ATTRS = ["Make_Code", "Product_Code", "Model_Variant"]


def _num(X: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column, or an all-NaN column when the field was not collected."""
    if col in X:
        return pd.to_numeric(X[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=X.index, dtype=float)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Cleaned raw frame -> model-ready feature frame (numeric + string categoricals).

    Parameters
    ----------
    salary_cap_quantile : upper winsorisation quantile for Net_salary (learned on fit).
    """

    def __init__(self, salary_cap_quantile: float = 0.995):
        self.salary_cap_quantile = salary_cap_quantile

    # ---- fit -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # Pincode -> Final_Tier is 1:1 in the data; learn it to fill missing tiers.
        if {"Pincode", "Final_Tier"} <= set(X.columns):
            pt = X.dropna(subset=["Pincode", "Final_Tier"])
            self.pin_tier_ = pt.groupby("Pincode")["Final_Tier"].agg(lambda s: s.mode().iat[0]).to_dict()
            p3 = pt.assign(p3=pt["Pincode"].str[:3])
            self.pin3_tier_ = p3.groupby("p3")["Final_Tier"].agg(lambda s: s.mode().iat[0]).to_dict()
        else:
            self.pin_tier_, self.pin3_tier_ = {}, {}
        # Model_Description -> make / product / variant (each description maps to one).
        if "Model_Description" in X:
            md = X.dropna(subset=["Model_Description"])
            self.vehicle_map_ = {
                col: md.dropna(subset=[col]).groupby("Model_Description")[col]
                .agg(lambda s: s.mode().iat[0]).to_dict()
                for col in VEHICLE_ATTRS if col in md
            }
        else:
            self.vehicle_map_ = {}
        sal = _num(X, "Net_salary")
        self.salary_cap_ = float(np.nanquantile(sal, self.salary_cap_quantile)) if sal.notna().any() else np.inf
        return self

    # ---- transform -------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        pin = X["Pincode"].astype("string") if "Pincode" in X else pd.Series(pd.NA, index=X.index, dtype="string")

        # Location. The learned pincode lookup takes priority over a supplied Final_Tier, so a
        # chatbot application (pincode only) and a training row get exactly the same value.
        # Supplied tier, then the 3-digit-prefix lookup, are fallbacks for unseen pincodes.
        empty = pd.Series(pd.NA, index=X.index, dtype="string")
        given_tier = X["Final_Tier"].astype("string") if "Final_Tier" in X else empty
        tier = (pin.map(self.pin_tier_).astype("string")
                .fillna(given_tier)
                .fillna(pin.str[:3].map(self.pin3_tier_).astype("string")))
        out["tier_ord"] = pd.to_numeric(tier.str[:2], errors="coerce").astype("float64")
        out["pin2"] = pin.str[:2].fillna(UNKNOWN)   # postal circle / state
        out["pin3"] = pin.str[:3].fillna(UNKNOWN)   # sorting district
        out["pincode"] = pin.fillna(UNKNOWN)

        # Vehicle: make / product / variant derived from the model description (same priority rule)
        for col in VEHICLE_ATTRS:
            given = X[col].astype("string") if col in X else empty
            derived = (X["Model_Description"].map(self.vehicle_map_[col]).astype("string")
                       if "Model_Description" in X and col in self.vehicle_map_ else empty)
            out[col] = derived.fillna(given).fillna(UNKNOWN)
        if "Model_Description" in X:
            out["Model_Description"] = X["Model_Description"].astype("string").fillna(UNKNOWN)

        # Money
        loan = _num(X, "Loan_Amount")
        ltv = _num(X, "LTV")
        sal = _num(X, "Net_salary").clip(upper=self.salary_cap_)
        out["Loan_Amount"] = loan
        out["LTV"] = ltv
        out["Net_salary"] = sal
        out["vehicle_value"] = loan / (ltv / 100.0)
        out["down_payment"] = out["vehicle_value"] - loan
        out["loan_to_income"] = loan / sal               # loan as multiple of monthly income
        out["salary_missing"] = sal.isna().astype(int)
        out["ltv_missing"] = ltv.isna().astype(int)

        # Applicant
        out["Age"] = _num(X, "Age")
        for col in ["Employment_Type", "PAST_LOANS_ACTIVE", "Qualifications", "Resident_Type", "Gender"]:
            if col in X:
                out[col] = X[col].astype("string").fillna(UNKNOWN)

        return out

    def get_feature_names_out(self, input_features=None):
        return None
