"""Model pipeline construction, threshold selection and artifact I/O."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from . import config as C
from .features import FeatureBuilder


@dataclass
class FeatureSpec:
    """Which engineered columns go into the model, by encoding type."""
    numeric: list[str]
    low_card: list[str]   # few levels: native categorical (HGB) / one-hot (LR)
    high_card: list[str]  # many levels: cross-fitted target encoding

    @property
    def all(self) -> list[str]:
        return self.numeric + self.low_card + self.high_card

    def drop(self, cols) -> "FeatureSpec":
        cols = set(cols)
        return FeatureSpec([c for c in self.numeric if c not in cols],
                           [c for c in self.low_card if c not in cols],
                           [c for c in self.high_card if c not in cols])


# Every engineered feature (used for EDA / ablation).
FULL_SPEC = FeatureSpec(
    numeric=["Age", "Loan_Amount", "LTV", "Net_salary", "vehicle_value", "down_payment",
             "loan_to_income", "tier_ord", "salary_missing", "ltv_missing"],
    low_card=["Employment_Type", "PAST_LOANS_ACTIVE", "Qualifications", "Resident_Type", "Gender",
              "Product_Code", "Make_Code", "Model_Variant", "pin2"],
    high_card=["pincode", "pin3", "Model_Description"],
)

# Final production spec, chosen in notebooks/02 (question-group ablation).
# Raw inputs needed: Pincode, Age, Employment_Type, PAST_LOANS_ACTIVE,
# Model_Description, Loan_Amount, LTV (= Loan_Amount / vehicle price).
FINAL_SPEC = FeatureSpec(
    numeric=["Age", "Loan_Amount", "LTV", "vehicle_value", "down_payment", "tier_ord", "ltv_missing"],
    low_card=["Employment_Type", "PAST_LOANS_ACTIVE", "Product_Code", "Make_Code", "Model_Variant", "pin2"],
    high_card=["pin3", "Model_Description"],
)

DEFAULT_HGB_PARAMS = dict(
    learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=100,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=50, random_state=C.RANDOM_STATE,
)


DEFAULT_RF_PARAMS = dict(
    n_estimators=500, max_features="sqrt", min_samples_leaf=50, n_jobs=-1, random_state=C.RANDOM_STATE,
)

DEFAULT_XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
    max_cat_to_onehot=1, eval_metric="logloss", n_jobs=-1, random_state=C.RANDOM_STATE,
)

DEFAULT_LGBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, min_data_per_group=50, cat_smooth=10,
    n_jobs=-1, random_state=C.RANDOM_STATE, verbose=-1,
)

MODEL_NAMES = ("logreg", "hgb", "rf", "xgb", "lgbm")


class ToCategorical(BaseEstimator, TransformerMixin):
    """Cast ordinal-encoded columns to pandas `category` with levels frozen at fit time.

    XGBoost (`enable_categorical`) and LightGBM pick up `category` columns as native
    categoricals. Freezing the levels keeps train and scoring frames consistent;
    unseen codes become NaN (treated as missing by both libraries).
    """

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        # Ordinal codes arrive as floats; XGBoost wants integer category levels.
        self.categories_ = {c: sorted(int(v) for v in pd.unique(X[c].dropna())) for c in self.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(pd.to_numeric(X[c]).round().astype("Int64"), categories=cats)
        return X


def _tree_encoder(spec: FeatureSpec) -> ColumnTransformer:
    """Numeric passthrough + ordinal codes for low-card + cross-fitted target encoding for high-card."""
    return ColumnTransformer(
        [("num", "passthrough", spec.numeric),
         ("low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1, min_frequency=20), spec.low_card),
         ("high", TargetEncoder(target_type="binary", random_state=C.RANDOM_STATE), spec.high_card)],
        verbose_feature_names_out=False,
    )


def build_pipeline(spec: FeatureSpec = FINAL_SPEC, model: str = "hgb", **params) -> Pipeline:
    """Feature engineering + encoding + classifier.

    model: "logreg" | "hgb" (HistGradientBoosting) | "rf" (RandomForest) |
           "xgb" (XGBoost) | "lgbm" (LightGBM).
    """
    n_num, n_low = len(spec.numeric), len(spec.low_card)
    if model == "rf":
        # RF has no native categoricals: low-card ordinal codes are used as numbers (fine for trees).
        clf = RandomForestClassifier(**{**DEFAULT_RF_PARAMS, **params})
        return Pipeline([("features", FeatureBuilder()), ("encode", _tree_encoder(spec)), ("clf", clf)])
    if model in ("xgb", "lgbm"):
        # Imported lazily so the chatbot / scoring code only needs these packages if that model is deployed.
        if model == "xgb":
            from xgboost import XGBClassifier
            clf = XGBClassifier(**{**DEFAULT_XGB_PARAMS, **params})
        else:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(**{**DEFAULT_LGBM_PARAMS, **params})
        enc = _tree_encoder(spec).set_output(transform="pandas")
        return Pipeline([("features", FeatureBuilder()), ("encode", enc),
                         ("cat", ToCategorical(spec.low_card)), ("clf", clf)])
    if model == "hgb":
        ct = ColumnTransformer(
            [("num", "passthrough", spec.numeric),
             ("low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                    encoded_missing_value=-1, min_frequency=20), spec.low_card),
             ("high", TargetEncoder(target_type="binary", random_state=C.RANDOM_STATE), spec.high_card)],
            verbose_feature_names_out=False,
        )
        hp = {**DEFAULT_HGB_PARAMS, **params}
        cat_idx = list(range(n_num, n_num + n_low)) or None
        clf = HistGradientBoostingClassifier(categorical_features=cat_idx, **hp)
    elif model == "logreg":
        ct = ColumnTransformer(
            [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), spec.numeric),
             ("low", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), spec.low_card),
             ("high", Pipeline([("te", TargetEncoder(target_type="binary", random_state=C.RANDOM_STATE)),
                                ("sc", StandardScaler())]), spec.high_card)],
            verbose_feature_names_out=False,
        )
        clf = LogisticRegression(max_iter=3000, **({"C": 1.0} | params))
    else:
        raise ValueError(model)
    return Pipeline([("features", FeatureBuilder()), ("encode", ct), ("clf", clf)])


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def expected_cost_curve(y_true, p_declined, cost_fn: float, cost_fp: float, thresholds=None) -> pd.DataFrame:
    """Cost of each threshold. Class 1 = DECLINED (risky).

    FN = risky applicant approved (credit loss, cost_fn)
    FP = good applicant declined (lost business, cost_fp)
    Decline when p_declined >= threshold.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p_declined)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.61, 0.005), 3)
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
        rows.append(dict(
            threshold=t, tp=tp, fp=fp, fn=fn, tn=tn,
            decline_rate=(tp + fp) / len(p),
            recall_declined=tp / max(tp + fn, 1), precision_declined=tp / max(tp + fp, 1),
            recall_approved=tn / max(tn + fp, 1), precision_approved=tn / max(tn + fn, 1),
            accuracy=(tp + tn) / len(p),
            cost=cost_fn * fn + cost_fp * fp,
        ))
    df = pd.DataFrame(rows)
    df["cost_per_app"] = df["cost"] / len(p)
    return df


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------
@dataclass
class ModelArtifact:
    pipeline: Pipeline
    threshold: float
    raw_input_fields: list[str]
    metadata: dict = field(default_factory=dict)

    def save(self, path=C.MODELS_DIR / "loan_decision_model.joblib"):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        meta = {**self.metadata, "threshold": self.threshold, "raw_input_fields": self.raw_input_fields}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return path

    @staticmethod
    def load(path=C.MODELS_DIR / "loan_decision_model.joblib") -> "ModelArtifact":
        return joblib.load(path)


def timed_predict(pipeline, X: pd.DataFrame, n_repeat: int = 200) -> np.ndarray:
    """Wall-clock latency (ms) of scoring X, repeated n_repeat times."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        pipeline.predict_proba(X)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)
