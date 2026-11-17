"""Chatbot configuration. Everything here can be overridden with environment
variables (or a `chat/.env` file – see `.env.example`)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CHAT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHAT_DIR.parent
load_dotenv(CHAT_DIR / ".env")

# Make `src/loan_model` importable (needed to unpickle the trained pipeline).
_SRC = str(PROJECT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---- LLM (interaction layer only – never makes the credit decision) --------
LLM_MODEL = os.getenv("CHAT_LLM_MODEL", "gemini-4-argon")        # Gemini 4 Argon: efficient frontier reasoning model
LLM_THINKING_LEVEL = os.getenv("CHAT_LLM_THINKING", "low")         # minimal | low | medium | high – low keeps chat replies fast
LLM_MAX_TOKENS = int(os.getenv("CHAT_LLM_MAX_TOKENS", "4000"))

# ---- ML model plug-in --------------------------------------------------------
# "auto" (default)                    -> trained model in models/loan_decision_model.joblib if present, else placeholder
# "trained"                           -> trained model (src/loan_model/predict.py LoanScorer); fail if missing
# "placeholder"                       -> stub model (no real decision)
# "sklearn:models/loan_model.joblib"  -> joblib-saved sklearn pipeline (path relative to project root)
# "my_package.my_module:MyModel"      -> any class implementing chat.ml_model.LoanModel
LOAN_MODEL = os.getenv("LOAN_MODEL", "auto")
TRAINED_MODEL_PATH = PROJECT_ROOT / "models" / "loan_decision_model.joblib"
# The trained model carries its own threshold (0.09, chosen in the model report). Set DECISION_THRESHOLD
# only to override it; the placeholder / generic sklearn adapters use 0.5 when it's unset.
DECISION_THRESHOLD_OVERRIDE = float(os.environ["DECISION_THRESHOLD"]) if os.getenv("DECISION_THRESHOLD") else None
DECISION_THRESHOLD = DECISION_THRESHOLD_OVERRIDE if DECISION_THRESHOLD_OVERRIDE is not None else 0.5

# ---- Audit log ---------------------------------------------------------------
AUDIT_LOG = Path(os.getenv("CHAT_AUDIT_LOG", PROJECT_ROOT / "logs" / "chat_audit.jsonl"))
