"""ML model plug-in point.

The chatbot only depends on the `LoanModel` protocol and the `ModelOutput`
dataclass below. The credit decision comes exclusively from here – the LLM
never makes it.

To plug in the trained model, either
  * save an sklearn pipeline with joblib and set  LOAN_MODEL=sklearn:models/loan_model.joblib
  * or write a class implementing `LoanModel` and set  LOAN_MODEL=my_pkg.my_module:MyModel
"""
from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, runtime_checkable

from langsmith import traceable

from . import audit, config

Decision = Literal["APPROVED", "DECLINED"]

# Fields the trained model consumes (mirrors src/loan_model/config.py MODEL_INPUT_FIELDS).
DEFAULT_REQUIRED_FIELDS = [
    "Pincode", "Age", "Employment_Type", "Net_salary", "PAST_LOANS_ACTIVE",
    "Model_Description", "Loan_Amount", "LTV",
]


@dataclass
class ModelOutput:
    """The contract every model must return."""
    decision: Decision                 # final call, from probability vs threshold
    probability_decline: float         # model's P(DECLINED), 0..1
    threshold: float                   # cut-off used: p >= threshold -> DECLINED
    model_name: str
    model_version: str
    features: dict = field(default_factory=dict)   # exact inputs the model scored (for the audit log)
    latency_ms: float = 0.0
    is_placeholder: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class LoanModel(Protocol):
    name: str
    version: str
    required_fields: list[str]    # raw fields the model needs (see chat/fields.py)

    def predict(self, application: dict) -> ModelOutput: ...


def decide(p_decline: float, threshold: float) -> Decision:
    return "DECLINED" if p_decline >= threshold else "APPROVED"


# ---------------------------------------------------------------------------
class PlaceholderModel:
    """BLANK MODEL – replace with the trained model. Returns a fixed score so the
    chat flow can be exercised end to end; output is flagged `is_placeholder`."""
    name = "placeholder"
    version = "0"
    required_fields = DEFAULT_REQUIRED_FIELDS

    def __init__(self, threshold: float = config.DECISION_THRESHOLD):
        self.threshold = threshold

    def predict(self, application: dict) -> ModelOutput:
        # TODO: replace with the real model's probability of DECLINED.
        p = 0.0
        return ModelOutput(
            decision=decide(p, self.threshold), probability_decline=p, threshold=self.threshold,
            model_name=self.name, model_version=self.version, features=dict(application),
            is_placeholder=True,
        )


class SklearnPipelineModel:
    """Adapter for a joblib-saved sklearn Pipeline whose `predict_proba[:, 1]`
    is P(DECLINED). The file may hold the pipeline itself, or a dict:
    {"pipeline": ..., "threshold": 0.42, "version": "v1", "required_fields": [...]}"""
    name = "sklearn-pipeline"

    def __init__(self, path: str, threshold: float | None = None):
        import joblib
        p = config.PROJECT_ROOT / path if not path.startswith("/") else path
        obj = joblib.load(p)
        if isinstance(obj, dict):
            self.pipeline = obj["pipeline"]
            self.threshold = threshold if threshold is not None else obj.get("threshold", config.DECISION_THRESHOLD)
            self.version = str(obj.get("version", p.name if hasattr(p, "name") else path))
            self.required_fields = list(obj.get("required_fields", DEFAULT_REQUIRED_FIELDS))
        else:
            self.pipeline = obj
            self.threshold = threshold if threshold is not None else config.DECISION_THRESHOLD
            self.version = str(path)
            self.required_fields = DEFAULT_REQUIRED_FIELDS

    def predict(self, application: dict) -> ModelOutput:
        import pandas as pd
        X = pd.DataFrame([application])
        p = float(self.pipeline.predict_proba(X)[0, 1])
        return ModelOutput(
            decision=decide(p, self.threshold), probability_decline=p, threshold=self.threshold,
            model_name=self.name, model_version=self.version, features=dict(application),
        )


class LoanScorerModel:
    """The trained ABC Credit model (models/loan_decision_model.joblib), scored through
    the project's own inference API `src/loan_model/predict.py::LoanScorer`, so the
    chatbot uses exactly the same cleaning, feature engineering and threshold as
    training and hold-out scoring."""
    name = "loan-decision-hgb"

    def __init__(self, path: str | None = None):
        from loan_model.predict import LoanScorer   # src/ is on sys.path (see config.py)

        self.scorer = LoanScorer(path) if path else LoanScorer()
        art = self.scorer.artifact
        self.required_fields = list(art.raw_input_fields)
        self.version = str(art.metadata.get("trained_at", "unknown"))
        if config.DECISION_THRESHOLD_OVERRIDE is not None:       # optional business override
            self.scorer.threshold = config.DECISION_THRESHOLD_OVERRIDE
        self.threshold = self.scorer.threshold

    def predict(self, application: dict) -> ModelOutput:
        d = self.scorer.score_application(application)
        return ModelOutput(
            decision=d.decision, probability_decline=d.p_declined, threshold=d.threshold,
            model_name=self.name, model_version=self.version,
            features={"inputs": dict(application), "engineered": d.features},
        )


def load_model(spec: str = config.LOAN_MODEL) -> LoanModel:
    if spec == "auto":
        return LoanScorerModel() if config.TRAINED_MODEL_PATH.exists() else PlaceholderModel()
    if spec == "trained":
        return LoanScorerModel()
    if spec == "placeholder":
        return PlaceholderModel()
    if spec.startswith("sklearn:"):
        return SklearnPipelineModel(spec.split(":", 1)[1])
    module, _, cls = spec.partition(":")
    model = getattr(importlib.import_module(module), cls)()
    if not isinstance(model, LoanModel):
        raise TypeError(f"{spec} does not implement LoanModel (name, version, required_fields, predict).")
    return model


@traceable(run_type="tool", name="ml_model.predict",
           process_inputs=lambda i: {"model": f"{i['model'].name}@{i['model'].version}",
                                     "application": audit.mask_fields(i["application"])})
def timed_predict(model: LoanModel, application: dict) -> ModelOutput:
    t0 = time.perf_counter()
    out = model.predict(application)
    out.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return out
