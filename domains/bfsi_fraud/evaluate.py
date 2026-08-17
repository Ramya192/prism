# domains/bfsi_fraud/evaluate.py
# Ported from fraud_detection/evaluate.py — logic unchanged, paths updated.
# "Honest evaluation on the untouched, naturally-imbalanced holdout."
# Run from the prism/ repo root: python domains/bfsi_fraud/evaluate.py
# Requires domains/bfsi_fraud/data/train.csv and test_holdout.csv — regenerate
# with prepare_data.py first (they're not committed; see that script's docstring).

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             confusion_matrix, precision_score, recall_score)
from domains.bfsi_fraud.tools.ml_scorer import MLScorer

_DATA = Path(__file__).parent / "data"

scorer = MLScorer(data_path=str(_DATA / "train.csv"))
test = pd.read_csv(_DATA / "test_holdout.csv")

probs = scorer.score_batch(test)
y = test["is_Fraud"].values

print(f"\nHoldout: {len(test):,} txns, {y.sum()} fraud ({y.mean():.4%} base rate)")
print(f"PR-AUC  : {average_precision_score(y, probs):.4f}   <- the headline metric")
print(f"ROC-AUC : {roc_auc_score(y, probs):.4f}   <- flattering, report both")

for t in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70):
    pred = [1 if p >= t else 0 for p in probs]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec = precision_score(y, pred, zero_division=0)
    print(f"\n  threshold {t}:  TP={tp}  FP={fp}  FN={fn}")
    print(f"    recall {recall_score(y, pred):.3f} | precision {prec:.4f} "
          f"| alerts per fraud {(tp+fp)/tp if tp else float('nan'):.1f}")

# ── Cost-based threshold: derive it from the money, not intuition ──────
fraud_amounts = test.loc[test.is_Fraud == 1, "Amount"]
print(f"\nfraud amount — mean {fraud_amounts.mean():.2f} "
      f"median {fraud_amounts.median():.2f} total {fraud_amounts.sum():.2f}")

COST_FP = 3.00     # analyst review time — substitute your own estimate

print()
for t in np.arange(0.02, 0.95, 0.02):
    pred = probs >= t
    missed = test.loc[pred == 0, :]
    fn_loss = missed.loc[missed.is_Fraud == 1, "Amount"].sum()
    fp_cost = ((pred == 1) & (test.is_Fraud == 0)).sum() * COST_FP
    print(f"  t={t:.2f}  fraud_lost={fn_loss:9.2f}  review_cost={fp_cost:8.2f}  total={fn_loss+fp_cost:9.2f}")
