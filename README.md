# ABC Credit: Loan Approval Model

The predictive part of the ABC Credit loan-approval chatbot project. A trained model takes 7 applicant answers and returns
**APPROVED / DECLINED** from its probability and a fixed, business-justified threshold. The chatbot (conversation layer) is a
separate phase and will call `LoanScorer` in `src/loan_model/predict.py`.

## Project structure

```
pgai/
├── data/
│   ├── training_data.csv                 # raw input (as provided, never modified)
│   └── processed/                        # written by notebook 02
│       ├── train_clean.csv               # 80% stratified split, cleaned
│       ├── test_clean.csv                # 20% internal test split, cleaned
│       └── cleaning_report.json          # what each cleaning rule changed
├── notebooks/
│   ├── 01_eda.ipynb                      # distributions, target balance, data-quality audit, field vs target
│   ├── 02_preprocessing_feature_selection.ipynb   # cleaning, split, features, question-level selection
│   └── 03_model_training_evaluation.ipynb         # model comparison, tuning, threshold, test metrics, latency, artifact
├── src/loan_model/                       # reusable package (used by notebooks, scripts and the future chatbot)
│   ├── config.py                         # paths and constants
│   ├── cleaning.py                       # deterministic row-wise cleaning
│   ├── features.py                       # FeatureBuilder (sklearn transformer: lookups, ratios, pincode prefixes)
│   ├── modeling.py                       # pipeline builder, feature specs, cost curve, ModelArtifact
│   └── predict.py                        # LoanScorer: batch and single-application scoring
├── models/
│   ├── loan_decision_model.joblib        # production pipeline + threshold
│   └── loan_decision_model.json          # metadata: params, features, threshold, metrics
├── reports/
│   ├── report.md                         # the short project report (fields, issues, model, metrics, X and Y)
│   ├── metrics.json
│   ├── internal_test_predictions.csv
│   └── figures/                          # every plot from the notebooks
├── scripts/
│   └── predict_holdout.py                # score the hold-out test set
├── requirements.txt
└── venv/
```

## Quick start

```bash
venv/bin/pip install -r requirements.txt
```

Run the notebooks in order (01 → 02 → 03). Notebook 02 writes the processed splits that 03 reads, and 03 writes the model.

```bash
cd notebooks && ../venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_preprocessing_feature_selection.ipynb 03_model_training_evaluation.ipynb
```

Score the hold-out set. The output has the columns `ID, P_DECLINED, Decision`, and metrics are printed if the file has labels:

```bash
venv/bin/python scripts/predict_holdout.py path/to/holdout.csv -o reports/holdout_predictions.csv
```

Score one application from Python, as the chatbot will:

```python
import sys; sys.path.insert(0, "src")
from loan_model.predict import LoanScorer
scorer = LoanScorer()
scorer.score_application({
    "Pincode": "560068", "Age": 29, "Employment_Type": "SAL", "PAST_LOANS_ACTIVE": "NO_PAST_LOANS",
    "Model_Description": "TVS RAIDER DISC BSVI", "Loan_Amount": 95000, "LTV": 88.5,   # LTV = loan / vehicle price * 100
})
# -> Decision(decision='DECLINED', p_declined=0.1389, threshold=0.09, latency_ms=..., features={...})
```

The returned `features` dict holds the engineered values, ready for the audit log (objective d).

## Results at a glance (internal 20% test set)

| | |
|---|---|
| Inputs (chatbot questions) | 7: pincode, vehicle model, vehicle price, loan amount, active past loans, age, employment type |
| Model | HistGradientBoostingClassifier (tuned) |
| ROC-AUC / PR-AUC | 0.786 / 0.184 (random PR-AUC = 0.053) |
| Threshold | P(DECLINED) ≥ 0.09 (10:1 cost of approving a risky applicant vs declining a good one) |
| DECLINED recall / precision | 0.506 / 0.151 |
| Accuracy | 0.824 |
| Latency, one application | p50 35 ms, p99 53 ms. Commitment Y ≤ 1 s |

See [reports/report.md](reports/report.md) for the reasoning behind each choice.
