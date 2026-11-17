# ABC Credit: Loan Approval Model Report

*Scope: data analysis, cleaning, feature selection, predictive model, threshold and latency. The chatbot layer is a separate
phase. This model is the decision engine it will call (`src/loan_model/predict.py`).*

All numbers are reproducible from `notebooks/01–03`. Figures are in `reports/figures/`, and metrics are in `reports/metrics.json`.

---

## 1. Data and key findings

* 108,423 rows × 17 columns. **5.26% DECLINED**, so the classes are heavily imbalanced. A model that approves everyone scores 94.7%
  accuracy and is useless, so we judge the model on ranking metrics (ROC-AUC, PR-AUC) and on precision and recall for the DECLINED class.
* **LTV is the dominant risk signal.** The decline rate rises steadily from 0.6% (LTV ≤ 60) to 14.7% (LTV > 90). On its own, LTV has a ROC-AUC of 0.74.
* **Geography matters.** Decline rates range from 1.3% to 12.9% across postal circles (first two digits of the pincode). The Final_Tier field captures part of this.
* The vehicle model, an active past loan (which *lowers* risk: 4.2% vs 6.2%), younger age and employment type each carry weaker signal.
* **Self-reported Net_salary carries no signal** (univariate AUC 0.51), and it has the worst data quality.
  Qualifications and Resident_Type add nothing either. Gender adds nothing and is a protected attribute in any case.

## 2. Data-quality issues and how they were handled

| Issue | Rows | Handling |
|---|---|---|
| Fully blank trailing row (no ID, no target) | 1 | Dropped |
| Duplicate records under different IDs (3 of them with conflicting labels) | 13 | Exact duplicates dropped; the conflicting pairs are kept as label noise |
| `Age` = `' '` | 1 | Set to missing; the model handles NaN natively |
| `LTV` = 0 (impossible for a financed vehicle) | 65 | Set to missing, plus an `ltv_missing` flag |
| `Net_salary` 0 or < ₹5,000/month | 887 | Set to missing (implausible); upper winsorisation at the 99.5th percentile (learned on train). Salary was later dropped from the model. |
| `Net_salary` up to ₹9.8L/month (probably annual figures) | 193 > ₹2L | Same winsorisation |
| Age 61–69, outside the 18–60 policy | 1,105 | Kept for training because they are real decisions. The chatbot enforces 18–60 as an eligibility rule before scoring. |
| Mojibake in `Model_Description` (`â€“`) | 5,028 (5,032 descriptions rewritten in total) | Encoding repaired; whitespace and dashes normalised |
| Inconsistent variant spelling (`125CC DISC` vs `125 CC DISC`, `XL 100CC`, `RTR 200CC`) | 20,013 | `NNN CC` format normalised |
| Missing categoricals (4–28 rows per field) | – | Coded as an explicit `UNKNOWN` level. Tier comes from a pincode lookup, falling back to the 3-digit prefix. Make, product and variant come from the model description. |
| Rare levels (`NONEARNMEM` 8 rows, `SCOOTY` 9, `165 CC` 1) | – | Grouped as infrequent (minimum frequency 20) |
| Unseen categories at scoring time | – | Every encoder uses `handle_unknown`. Tested with unseen pincodes, employment types and models, and with blank and garbage inputs. |

The cleaning is row-wise and deterministic (`cleaning.py`). Everything that is learned from data, such as lookups, caps and target
encodings, lives inside the sklearn `Pipeline` and is fitted on training folds only, so nothing leaks from test data.

## 3. Feature engineering and selection: which fields we kept and why

Each chatbot question fills one *group* of features, so selection was done per question. We ran a drop-one ablation and forward
selection with 3-fold CV ROC-AUC on the training split (`figures/11_forward_selection.png`):

| Questions | CV AUC |
|---|---|
| Loan amount + vehicle price (→ LTV) | 0.735 |
| + Pincode | 0.764 |
| + Vehicle model | 0.771 |
| + Active past loans | 0.774 |
| + Age | 0.776 |
| **+ Employment type (final: 7 questions)** | **0.778** |
| + Resident type, qualifications, net salary (10 questions) | 0.778 |

**Kept (7 inputs):** Pincode, Vehicle model, Vehicle price, Loan amount, Active past loans, Age, Employment type.
**Engineered:** `LTV` and `vehicle_value`/`down_payment` from loan and price; `tier_ord`, `pin2` (postal circle) and `pin3` (district)
from the pincode; make, product and variant from the model; `ltv_missing`.
**Dropped:** Net_salary, Qualifications and Resident_Type (no measurable gain), the full 6-digit pincode (no gain over its prefixes,
and its 14k levels are sparse), and Gender (protected attribute).

Encoding: low-cardinality fields use native categorical splits in the gradient-boosting model. `pin3` and `Model_Description` use
cross-fitted, smoothed target encoding.

## 4. Model choice

5-fold stratified CV on the training split (80%):

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Always APPROVE | 0.500 | 0.053 | 0.053 |
| Logistic regression | 0.780 | 0.175 | 0.0465 |
| Logistic regression, balanced class weights | 0.779 | 0.170 | 0.198 |
| HistGradientBoosting (defaults) | 0.782 | 0.176 | 0.0465 |
| **HistGradientBoosting (tuned)** | **0.784** | **0.179** | – |

We chose **HistGradientBoostingClassifier**, tuned with a 20-trial randomised search (learning rate 0.026, 15 leaves, minimum 74
samples per leaf, L2 0.21, 50% feature sampling). It is the best model on every fold, handles missing values natively, fits the
non-linear LTV effect, and its probabilities are well calibrated. Imbalance is handled at the threshold rather than through
re-weighting, which would have destroyed calibration.

The production artifact (`models/loan_decision_model.joblib`) uses the same settings refit on all 108,409 labelled rows.

## 5. Threshold and its justification

The model outputs P(DECLINED). The decision rule is to **DECLINE when P ≥ 0.09**.

* Cost assumption: a risky applicant who is wrongly approved costs about ₹45k (roughly half of a ₹90k loan lost after repossession).
  A good applicant who is wrongly declined costs about ₹4.5k of lost margin. That gives a **10 : 1** cost ratio.
* The threshold minimises expected cost on out-of-fold training predictions. It matches the theoretical value 1/(1+10) ≈ 0.09
  because the model is calibrated.
* Sensitivity (out-of-fold): at 5:1 the threshold is 0.175 and recall on DECLINED is 23%. At 15:1 it is 0.065, recall is 67%, and 25% of good
  applicants are declined. The threshold is one value stored in the model metadata, so risk management can change it without retraining.

## 6. Evaluation on the internal test set (20% stratified, scored once)

| Ranking metric (threshold-free) | Value |
|---|---|
| ROC-AUC | **0.786** |
| PR-AUC (random = 0.053) | **0.184** |
| Brier score | 0.046 |

| At threshold 0.09 | DECLINED | APPROVED |
|---|---|---|
| Precision | 0.151 | 0.969 |
| **Recall** | **0.506** | 0.842 |
| F1 | 0.233 | 0.901 |

Accuracy 0.824 · balanced accuracy 0.674 · decline rate 17.6%

Confusion matrix (rows = actual, columns = predicted):

| | Pred. APPROVED | Pred. DECLINED |
|---|---|---|
| **Actual APPROVED** | 17,296 | 3,246 |
| **Actual DECLINED** | 563 | 577 |

For comparison, the default threshold of 0.5 gives 94.7% accuracy but catches **0** risky applicants.
Lift: the riskiest 10% of applicants by score contain 37% of all declines (3.7× lift). The safest 30% contain under 4%.

*Note:* ranking performance of about 0.78 AUC is close to what the given fields allow. The ablation shows a plateau, and adding every
remaining field gains nothing. Precision on DECLINED stays modest because the historical decline decisions depend on information
this dataset doesn't contain, such as bureau data.

## 7. Latency (Y)

End-to-end scoring of one application (raw answers → cleaning → features → model → decision), measured on an Apple-silicon laptop:
**p50 ≈ 35 ms, p95 ≈ 48 ms, p99 ≈ 53 ms.**
**Y = 1 second** (from final input to decision). The model uses under 6% of that budget, and the rest covers logging, network and
response rendering.

## 8. Completion time (X): provisional, to be confirmed in the chatbot phase

The model needs only **7 answers**. Pincode, age, employment type and active loans are single short answers. The vehicle model is
chosen from a list, and price and loan amount are two numbers. At roughly 15–20 seconds per answer, including reading and
validation, plus consent and the result, a session takes about 2–3 minutes. **Proposed X = 3 minutes.** That is short enough for a
dealership counter, where shorter forms mean fewer drop-offs, and it is achievable because the feature selection cut 10 candidate
questions to 7 without losing accuracy.

## 9. Reproducing

```
venv/bin/pip install -r requirements.txt
cd notebooks && ../venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_preprocessing_feature_selection.ipynb 03_model_training_evaluation.ipynb
venv/bin/python scripts/predict_holdout.py <holdout.csv>
```
