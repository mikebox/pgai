"""Model pipeline construction, threshold selection and artifact I/O."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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


def build_pipeline(spec: FeatureSpec = FINAL_SPEC, model: str = "hgb", **params) -> Pipeline:
    n_num, n_low = len(spec.numeric), len(spec.low_card)
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
