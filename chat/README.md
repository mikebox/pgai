# ABC Credit loan chatbot

LangGraph chatbot that collects an application, asks clarifying questions with Gemini (`gemini-4-argon`), and gets the decision from a separate ML model. The LLM only gathers inputs and talks to the applicant. It never makes the credit decision.

```
START ─┬─> closed                     already decided
       └─> extract ─┬─> ask           missing / invalid / unclear → Gemini asks
                    ├─> confirm       all collected → summary, "reply yes"
                    └─> predict       confirmed → ML model → APPROVED / DECLINED
```

The chat asks exactly what the trained model uses (7 questions), in 2 steps, each shown as a form card:
1. **About you:** age, pincode, employment type, existing loans.
2. **Your vehicle:** model (with suggestions from the 87 known models), on-road price, loan amount. LTV is computed as loan ÷ price.

Monthly income isn't asked because the model doesn't use it. A step that would hold a single question is folded into the step before it.

Then a review card with **Submit** / **Edit** buttons follows. Form answers are validated in code without an LLM call, which keeps each step fast. Gemini is called only when the applicant types free text, picks "Other…", or asks a question. Groups are defined in `fields.GROUPS`.

| File | Role |
|---|---|
| `graph.py` | LangGraph state, nodes, routing, `LoanChatbot` façade |
| `llm.py` | Gemini calls (Interactions API): structured extraction and the next reply |
| `fields.py` | Field registry: descriptions, validation, derived fields (LTV = loan / price) |
| `ml_model.py` | **ML plug-in point**: the `LoanModel` protocol, the `ModelOutput` contract and the placeholder model |
| `audit.py` | JSONL audit log (`logs/chat_audit.jsonl`) with PII masking |
| `observability.py` | LangSmith tracing (optional), with the same PII masking |
| `server.py` + `static/index.html` | Local web chat (Python standard library only, no accounts) |
| `cli.py` | Terminal chat |

## Run

```bash
pip install -r chat/requirements.txt
cp chat/.env.example chat/.env      # add GEMINI_API_KEY
python -m chat.server               # open http://localhost:8000   (or: python -m chat.cli)
python -m chat.tests.test_flow      # offline test with a fake LLM
```

## The ML model

By default (`LOAN_MODEL=auto`) the chat uses the trained model in `models/loan_decision_model.joblib`. It is the best model from `notebooks/03` (currently a tuned LightGBM classifier), and the chat calls it through `src/loan_model/predict.py` (`LoanScorer`), so cleaning, feature engineering and the threshold stored in the artifact (currently **0.095**) are exactly those used in training and hold-out scoring. The questions are driven by the model's `raw_input_fields`. The log records the inputs, the engineered features, P(DECLINED), the decision and the latency for every application.

## Plugging in a different model

Every model returns `ModelOutput`:

```python
decision: "APPROVED" | "DECLINED"   # P(DECLINED) >= threshold -> DECLINED
probability_decline: float
threshold: float
model_name, model_version: str
features: dict                      # exact inputs scored (logged)
latency_ms: float
is_placeholder: bool
```

You can plug the model in either of two ways:

1. **sklearn pipeline:** `joblib.dump({"pipeline": pipe, "threshold": 0.42, "version": "v1"}, "models/loan_model.joblib")`, then set `LOAN_MODEL=sklearn:models/loan_model.joblib`. The pipeline receives a one-row DataFrame with the raw fields (`Pincode, Age, Employment_Type, Net_salary, PAST_LOANS_ACTIVE, Model_Description, Loan_Amount, LTV`), and `predict_proba[:, 1]` must be P(DECLINED).
2. **Custom class:** implement `name`, `version`, `required_fields` and `predict(dict) -> ModelOutput`, then set `LOAN_MODEL=my_pkg.my_module:MyModel`.

The chatbot asks for whatever the model lists in `required_fields`. If the model needs a field that isn't asked yet, add a `FieldSpec` (or `DerivedField`) in `fields.py`.

## Observability (LangSmith)

Tracing is off by default. To turn it on, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `chat/.env`. Each applicant message then becomes one trace in the `abc-credit-loan-chatbot` project:

- **LangGraph run:** every node, with state in and out.
- **Each Gemini call:** prompt, reply, token usage (including thinking tokens) and latency.
- **ML model call:** inputs, probability, decision and latency.

Traces carry `session_id` metadata, so the **Threads** tab shows each application as one conversation. Every payload goes through the audit log's PII masker before it is sent.
