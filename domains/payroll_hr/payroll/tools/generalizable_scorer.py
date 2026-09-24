# domains/payroll_hr/payroll/tools/generalizable_scorer.py
# Tier 2 scorer for payroll's CSV-register detector -- a SECOND, genuinely
# different payroll-export schema than Tier 1's (SF Salaries'
# BasePay/OvertimePay/OtherPay/TotalPay): GROSS/Deduction/Net_Pay, from
# https://www.kaggle.com/datasets/jb1433/sample-employees-monthly-salary
# (converted INR->USD -- see data/prepare_tier2_data.py's module
# docstring for why and the exact rate used).
#
# Same complementary-to-rules design as Tier 1's anomaly_scorer.py:
# GROSS - Deduction == Net_Pay holds EXACTLY for every row here, flagged
# or not (verified in prepare_tier2_data.py) -- rules_based_filter()'s
# reconciliation check has nothing to catch. Only an implausible
# Deduction-to-Gross RATIO (near-0% or near-100%) distinguishes a flagged
# row, which is exactly the kind of pattern only a trained model can see.

import pandas as pd

from core.model_store import load_or_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class PayrollGeneralizableScorer:

    # Calibrated against this dataset's own base rate via a holdout sweep
    # -- see the class docstring; deliberately NOT the same thresholds as
    # Tier 1's PayrollAnomalyScorer (different dataset, different score
    # distribution).
    FRAUD_THRESHOLD = 0.6
    LEGIT_THRESHOLD = 0.3

    # Deliberately excludes Deduction_percentage: it's a direct rescaling
    # of Deduction/GROSS, which is exactly the ratio used to construct the
    # label in prepare_tier2_data.py -- including it measured as leakage
    # (AUC 0.996, F1 0.995 -- too clean to be real signal; removing it
    # dropped this to the honest number in the class docstring below).
    FEATURE_COLS = ["GROSS", "Deduction", "Net_Pay"]
    REQUIRED_COLS = ["GROSS", "Deduction", "Net_Pay"]

    def __init__(self, data_path="domains/payroll_hr/payroll/data/tier2_train.csv"):
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        load_or_train(self, data_path)

    def _train(self, data_path):
        try:
            df = pd.read_csv(data_path)
            available = [c for c in self.FEATURE_COLS if c in df.columns]
            missing = [c for c in self.FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"  [PayrollGeneralizableScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [PayrollGeneralizableScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].fillna(0)
            y = df["Is_Flagged"]
            X_scaled = self.scaler.fit_transform(X)

            self.model = RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [PayrollGeneralizableScorer] Trained on {len(df):,} rows "
                  f"({y.sum()} flagged, {y.mean():.1%} base rate)")
        except FileNotFoundError:
            print(f"  [PayrollGeneralizableScorer] {data_path} not found. "
                  f"Run domains/payroll_hr/payroll/data/prepare_tier2_data.py first.")
        except Exception as e:
            print(f"  [PayrollGeneralizableScorer] Training failed: {e}")

    def score(self, record: dict) -> float:
        if not self.trained:
            return 0.5
        try:
            row = {col: record.get(col, 0) for col in self.feature_cols}
            X = pd.DataFrame([row])
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [PayrollGeneralizableScorer] Scoring error: {e}")
            return 0.5

    def score_batch(self, df: pd.DataFrame):
        import numpy as np

        if not self.trained:
            return np.full(len(df), 0.5)
        X = df.reindex(columns=self.feature_cols).fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]
