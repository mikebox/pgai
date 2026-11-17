"""
ABC Credit - two-wheeler loan approval: reusable cleaning + feature engineering.

Everything the production pipeline needs lives here (not in a notebook) so that a
pipeline saved with joblib can be loaded again by the chatbot service:

    import sys; sys.path.append("<path to pilot folder>")
    import joblib, credit_pipeline          # makes the custom classes importable
    bundle = joblib.load("artifacts/approval_model.joblib")

The cleaning rules mirror pilot/data_cleaning.ipynb (Layer 1 = basic_clean,
business-aware fills = DomainImputer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Raw schema (what the chatbot / dealer system sends)
# --------------------------------------------------------------------------------------
RAW_COLUMNS = ["ID", "Qualifications", "Employment_Type", "Age", "Pincode", "Gender", "Product_Code",
               "Loan_Amount", "LTV", "Resident_Type", "Net_salary", "Final_Tier", "Make_Code",
               "Model_Description", "Model_Variant", "PAST_LOANS_ACTIVE"]
NUMERIC_RAW = ["Age", "Loan_Amount", "LTV", "Net_salary"]
TEXT_RAW = [c for c in RAW_COLUMNS if c not in NUMERIC_RAW]
TARGET = "Decision"

# Where each raw field comes from in the live journey
CUSTOMER_QUESTIONS = ["Qualifications", "Employment_Type", "Age", "Pincode", "Gender",
                      "Resident_Type", "Net_salary", "PAST_LOANS_ACTIVE"]
DEAL_INPUTS = ["Loan_Amount", "LTV"]                         # from the dealer's quote
CATALOGUE_INPUTS = ["Product_Code", "Make_Code", "Model_Description", "Model_Variant"]
LOOKUP_INPUTS = ["Final_Tier"]                                # looked up from pincode

# --------------------------------------------------------------------------------------
# Business constants (data dictionary)
# --------------------------------------------------------------------------------------
AGE_POLICY_MIN, AGE_POLICY_MAX, AGE_PLAUSIBLE_MAX = 18, 60, 100
NON_EARNER_TYPES = {"STU", "NONEARNMEM"}
STABLE_INCOME_TYPES = {"SAL", "PEN"}
MIN_PLAUSIBLE_SALARY = 1000
TIER_ORDER = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6, "07": 7}
PRODUCT_TO_MAKE = {"MO": ("MOPEDS", "XL 100 CC"), "EB": ("TVS", "EBIKE")}


# ======================================================================================
# Layer 1: rule-based cleaning (stateless, never drops rows)
# ======================================================================================
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-by-row cleaning. Safe on train, test or a single chatbot record."""
    df = df.copy()
    for c in RAW_COLUMNS:                       # chatbot may not send every field
        if c not in df.columns:
            df[c] = np.nan

    # 1) Text hygiene: everything text-like -> trimmed string, blanks -> missing
    for c in TEXT_RAW:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    df["Pincode"] = df["Pincode"].str.replace(r"\.0$", "", regex=True)

    # 2) Encoding glitch + case + 'CC' spacing
    for c in ["Model_Description", "Model_Variant", "Make_Code"]:
        df[c] = df[c].str.replace("â€“", "-", regex=False).str.upper()
    df["Model_Variant"] = df["Model_Variant"].str.replace(r"(\d+)\s*CC", r"\1 CC", regex=True)

    # 2b) Moped / e-bike map to a single make & variant
    for prod, (make, variant) in PRODUCT_TO_MAKE.items():
        is_prod = (df["Product_Code"] == prod).fillna(False).astype(bool)
        df.loc[is_prod & df["Make_Code"].isna(), "Make_Code"] = make
        df.loc[is_prod & df["Model_Variant"].isna(), "Model_Variant"] = variant

    # 3) Numbers
    for c in NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c].astype("string").str.strip(), errors="coerce").astype(float)

    # 4) Age: implausible -> missing; over-policy kept and flagged
    df.loc[(df["Age"] < AGE_POLICY_MIN) | (df["Age"] > AGE_PLAUSIBLE_MAX), "Age"] = np.nan
    df["age_above_policy"] = (df["Age"] > AGE_POLICY_MAX).astype(int)

    # 5) LTV must be in (0, 100]
    df["ltv_invalid"] = (~df["LTV"].between(0, 100, inclusive="right") & df["LTV"].notna()).astype(int)
    df.loc[df["ltv_invalid"] == 1, "LTV"] = np.nan

    # 6) Salary: 0 valid for non-earners, < Rs 1,000 for earners is a typo
    is_non_earner = df["Employment_Type"].isin(NON_EARNER_TYPES).fillna(False).astype(bool)
    bad_salary = ((df["Net_salary"] < MIN_PLAUSIBLE_SALARY) & ~is_non_earner).fillna(False).astype(bool)
    df["salary_invalid"] = bad_salary.astype(int)
    df.loc[bad_salary | (df["Net_salary"] < 0), "Net_salary"] = np.nan
    df["is_non_earner"] = is_non_earner.astype(int)

    # 7) Pincode stays text; validate; postal hierarchy
    valid_pin = df["Pincode"].str.fullmatch(r"[1-9]\d{5}").fillna(False).astype(bool)
    df["Pincode"] = df["Pincode"].where(valid_pin, pd.NA)
    df["pin_zone"] = df["Pincode"].str[:1]
    df["pin_district"] = df["Pincode"].str[:3]

    # 8) Final_Tier -> ordinal 1..7 (unknown labels -> missing, filled by DomainImputer)
    df["tier_rank"] = df["Final_Tier"].str[:2].map(TIER_ORDER).astype(float)

    # 9) Categoricals as plain object with np.nan (what scikit-learn expects)
    for c in TEXT_RAW + ["pin_zone", "pin_district"]:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df


def training_row_filter(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """TRAINING ONLY: drop empty rows, rows without a Decision, and exact duplicates."""
    n0 = len(df)
    inputs = [c for c in df.columns if c not in ("ID", TARGET)]
    stripped = df[inputs].apply(lambda s: s.astype("string").str.strip().replace("", pd.NA))
    df = df[stripped.notna().any(axis=1)]
    n1 = len(df)
    df = df[df[TARGET].astype("string").str.strip().isin(["APPROVED", "DECLINED"])]
    n2 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    if verbose:
        print(f"rows: start {n0:,} | empty -{n0 - n1} | no Decision -{n1 - n2} | "
              f"duplicates -{n2 - len(df)} | kept {len(df):,}")
    return df.reset_index(drop=True)


class RawCleaner(BaseEstimator, TransformerMixin):
    """sklearn wrapper around basic_clean (stateless)."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):          # stateless: always ready
        return True

    def transform(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        return basic_clean(pd.DataFrame(X))


# ======================================================================================
# Business-aware filling, learned from training rows only
# ======================================================================================
class DomainImputer(BaseEstimator, TransformerMixin):
    """LTV = 0/missing -> loan / typical bike price; missing tier -> tier of nearby pincodes."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        ok = X["LTV"].notna() & X["Loan_Amount"].notna()
        price = X.loc[ok, "Loan_Amount"] / (X.loc[ok, "LTV"] / 100)
        self.price_by_model_ = price.groupby(X.loc[ok, "Model_Description"]).median()
        self.price_by_make_ = price.groupby(X.loc[ok, "Make_Code"]).median()
        self.price_by_product_ = price.groupby(X.loc[ok, "Product_Code"]).median()
        self.price_overall_ = float(price.median())
        t = X.dropna(subset=["tier_rank"])
        mode = lambda s: s.mode().iloc[0]
        self.tier_by_district_ = t.dropna(subset=["pin_district"]).groupby("pin_district")["tier_rank"].agg(mode)
        self.tier_by_zone_ = t.dropna(subset=["pin_zone"]).groupby("pin_zone")["tier_rank"].agg(mode)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        need = X["LTV"].isna() & X["Loan_Amount"].notna()
        price = (X["Model_Description"].map(self.price_by_model_)
                 .fillna(X["Make_Code"].map(self.price_by_make_))
                 .fillna(X["Product_Code"].map(self.price_by_product_))
                 .fillna(self.price_overall_))
        X.loc[need, "LTV"] = (100 * X.loc[need, "Loan_Amount"] / price[need]).clip(1, 100)
        miss = X["tier_rank"].isna()
        guess = X["pin_district"].map(self.tier_by_district_).fillna(X["pin_zone"].map(self.tier_by_zone_))
        X.loc[miss, "tier_rank"] = guess[miss].astype(float)
        X["tier_imputed"] = miss.astype(int)
        return X


# ======================================================================================
# Feature engineering (stateless)
# ======================================================================================
def _flag(condition: pd.Series, source: pd.Series) -> pd.Series:
    """1/0 flag that stays missing when the source answer is missing."""
    return pd.Series(np.where(source.isna(), np.nan, condition.astype(float)), index=source.index)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Credit-risk features. EMI assumes `tenure_months` and `annual_rate` (reducing balance)."""

    AGE_BINS = [17, 25, 35, 45, 60, 200]
    AGE_LABELS = ["18-25", "26-35", "36-45", "46-60", "60+"]

    def __init__(self, tenure_months: int = 24, annual_rate: float = 0.22, high_ltv: float = 90.0):
        self.tenure_months = tenure_months
        self.annual_rate = annual_rate
        self.high_ltv = high_ltv

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):          # stateless: always ready
        return True

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        salary = X["Net_salary"].where(X["Net_salary"] > 0)        # 0 income -> ratios undefined
        ltv = X["LTV"].where(X["LTV"] > 0)

        X["Loan_to_Income"] = X["Loan_Amount"] / salary             # months of salary borrowed
        X["Vehicle_Price"] = X["Loan_Amount"] / (ltv / 100)
        X["Down_Payment"] = (X["Vehicle_Price"] - X["Loan_Amount"]).clip(lower=0)
        r, n = self.annual_rate / 12, self.tenure_months
        X["Approx_EMI"] = X["Loan_Amount"] * r * (1 + r) ** n / ((1 + r) ** n - 1)
        X["EMI_to_Income"] = X["Approx_EMI"] / salary               # FOIR-style burden
        X["Residual_Income"] = X["Net_salary"] - X["Approx_EMI"]    # money left after EMI
        X["Down_Payment_to_Income"] = X["Down_Payment"] / salary    # savings capacity
        X["Age_Band"] = pd.cut(X["Age"], self.AGE_BINS, labels=self.AGE_LABELS).astype(object)
        X["Age_Band"] = X["Age_Band"].where(X["Age_Band"].notna(), np.nan)
        X["Is_First_Time_Borrower"] = _flag(X["PAST_LOANS_ACTIVE"] == "NO_PAST_LOANS", X["PAST_LOANS_ACTIVE"])
        X["High_LTV"] = _flag(X["LTV"] >= self.high_ltv, X["LTV"])
        X["Is_Stable_Income"] = _flag(X["Employment_Type"].isin(STABLE_INCOME_TYPES), X["Employment_Type"])
        X["Owns_Home"] = _flag(X["Resident_Type"] == "O", X["Resident_Type"])
        X["Is_EV"] = _flag(X["Product_Code"] == "EB", X["Product_Code"])

        num = X.select_dtypes(include="number").columns
        X[num] = X[num].replace([np.inf, -np.inf], np.nan)
        return X


# ======================================================================================
# Model-ready encoding
# ======================================================================================
class QuantileCapper(BaseEstimator, TransformerMixin):
    """Winsorise at percentiles learned from training data."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X).astype(float)
        self.lo_, self.hi_ = X.quantile(self.lower).values, X.quantile(self.upper).values
        return self

    def transform(self, X):
        return np.clip(pd.DataFrame(X).astype(float).values, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(skewed=(), plain=(), flags=(), ordinal=(), onehot=(), target_enc=(),
                       scale: bool = False, min_frequency: int = 30) -> ColumnTransformer:
    """ColumnTransformer for one feature set. `scale=True` only for linear models."""
    def num_steps(log: bool):
        steps = []
        if log:
            steps += [("cap", QuantileCapper()),
                      ("log", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one"))]
        steps += [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps += [("scale", StandardScaler())]
        return Pipeline(steps)

    parts = []
    if skewed:
        parts.append(("skew", num_steps(True), list(skewed)))
    if plain:
        parts.append(("num", num_steps(False), list(plain)))
    if flags:
        parts.append(("flag", SimpleImputer(strategy="most_frequent"), list(flags)))
    if ordinal:
        ord_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
        parts.append(("ord", Pipeline(ord_steps), list(ordinal)))
    if onehot:
        parts.append(("ohe", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                  sparse_output=False))]), list(onehot)))
    if target_enc:
        te_steps = [("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("te", TargetEncoder(target_type="binary", smooth="auto",
                                         cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)))]
        if scale:
            te_steps.append(("scale", StandardScaler()))
        parts.append(("te", Pipeline(te_steps), list(target_enc)))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def processing_pipeline(preprocessor: ColumnTransformer, tenure_months=24, annual_rate=0.22) -> Pipeline:
    """Raw applicant record(s) -> model-ready matrix."""
    return Pipeline([
        ("clean", RawCleaner()),
        ("domain", DomainImputer()),
        ("features", FeatureEngineer(tenure_months=tenure_months, annual_rate=annual_rate)),
        ("encode", preprocessor),
    ])


# ======================================================================================
# Helpers
# ======================================================================================
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Association between two categoricals (0 = none, 1 = one fully determines the other)."""
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(x.fillna("Missing"), y.fillna("Missing"))
    if min(tab.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    # bias-corrected (Bergsma 2013)
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_c - 1, r_c - 1))))
