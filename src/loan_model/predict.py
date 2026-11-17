"""Inference API – what the (future) chatbot and the hold-out scoring script call.

The credit decision is made here, by the trained model + fixed threshold only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .cleaning import clean_raw
from .modeling import ModelArtifact


@dataclass
class Decision:
    decision: str             # "APPROVED" | "DECLINED"
    p_declined: float         # model probability of the DECLINED class
    threshold: float
    latency_ms: float
    features: dict            # engineered feature values (for audit logging)


class LoanScorer:
    def __init__(self, artifact_path=C.MODELS_DIR / "loan_decision_model.joblib"):
        self.artifact = ModelArtifact.load(artifact_path)
        self.pipeline = self.artifact.pipeline
        self.threshold = self.artifact.threshold

    def score_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring of raw (uncleaned) rows. Returns ID, P_DECLINED, Decision."""
        clean = clean_raw(raw)
        p = self.pipeline.predict_proba(clean)[:, 1]
        out = pd.DataFrame({"P_DECLINED": p.round(6)}, index=raw.index)
        out["Decision"] = np.where(p >= self.threshold, "DECLINED", "APPROVED")
        if C.ID_COL in raw:
            out.insert(0, C.ID_COL, raw[C.ID_COL].values)
        return out

    def score_application(self, application: dict) -> Decision:
        """Score one application given as {field: value} (missing fields allowed)."""
        t0 = time.perf_counter()
        raw = pd.DataFrame([{k: (None if v is None else str(v)) for k, v in application.items()}])
        clean = clean_raw(raw)
        feats = self.pipeline.named_steps["features"].transform(clean)
        p = float(self.pipeline[1:].predict_proba(feats)[0, 1])   # reuse the engineered features
        latency = (time.perf_counter() - t0) * 1000
        return Decision(
            decision="DECLINED" if p >= self.threshold else "APPROVED",
            p_declined=round(p, 6), threshold=self.threshold, latency_ms=round(latency, 2),
            features={k: (v.item() if hasattr(v, "item") else v) for k, v in feats.iloc[0].to_dict().items()},
        )
