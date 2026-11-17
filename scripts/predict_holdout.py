"""Score the hold-out test set with the trained model.

Usage (from the project root):
    venv/bin/python scripts/predict_holdout.py path/to/holdout.csv [-o reports/holdout_predictions.csv]

Input:  CSV with the same columns as training_data.csv (Decision may be absent).
Output: CSV with ID, P_DECLINED, Decision (APPROVED / DECLINED).
If the input contains a Decision column, evaluation metrics are printed as well.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score, roc_auc_score)

from loan_model import config as C  # noqa: E402
from loan_model.cleaning import load_raw  # noqa: E402
from loan_model.predict import LoanScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("-o", "--output", default=str(C.REPORTS_DIR / "holdout_predictions.csv"))
    args = ap.parse_args()

    raw = load_raw(args.input_csv)
    raw = raw[~raw.isna().all(axis=1)]          # ignore fully blank lines
    scorer = LoanScorer()
    preds = scorer.score_frame(raw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"Scored {len(preds):,} applications (threshold {scorer.threshold}) -> {args.output}")
    print(preds["Decision"].value_counts().to_string())

    if C.TARGET in raw and raw[C.TARGET].notna().any():
        mask = raw[C.TARGET].str.strip().str.upper().isin(["APPROVED", "DECLINED"])
        y = (raw.loc[mask, C.TARGET].str.strip().str.upper() == "DECLINED").astype(int)
        p = preds.loc[mask, "P_DECLINED"]
        yhat = (preds.loc[mask, "Decision"] == "DECLINED").astype(int)
        print("\nEvaluation against provided labels (positive class = DECLINED)")
        print(f"  ROC-AUC   {roc_auc_score(y, p):.4f}   PR-AUC {average_precision_score(y, p):.4f}")
        print(f"  Accuracy  {accuracy_score(y, yhat):.4f}")
        print(f"  Precision {precision_score(y, yhat, zero_division=0):.4f}   Recall {recall_score(y, yhat):.4f}   "
              f"F1 {f1_score(y, yhat):.4f}")
        print("  Confusion matrix [rows=actual APPROVED/DECLINED, cols=predicted]:")
        print("  ", confusion_matrix(y, yhat).tolist())


if __name__ == "__main__":
    main()
