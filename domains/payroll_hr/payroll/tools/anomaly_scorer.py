# domains/payroll_hr/payroll/tools/anomaly_scorer.py
# ML anomaly tier for CSV payroll-register uploads (many employee rows at
# once) -- see agents/payroll_detector_agent.py's module docstring for how
# this fits alongside the rules layer (reconciliation math) and the LLM
# fallback, mirroring banking fraud's rules -> ML -> LLM shape.
#
# Trained on REAL San Francisco city employee compensation
# (https://www.kaggle.com/datasets/kaggle/sf-salaries, CC0) with
# SYNTHETIC injected reconciliation-error labels -- see
# data/prepare_tier_ml_data.py's module docstring for why: no labeled
# payroll-fraud dataset exists publicly (real payroll data is
# confidential PII), and 100% of this dataset's real rows already
# reconcile exactly, so every positive label here is a deliberate,
# known-ground-truth injection, not a real observed fraud case.
#
# Honest caveat worth keeping in mind: reconciliation is exact arithmetic
# (stated_total == base+overtime+other), so a model given the raw
# components can only ever be learning to approximate that same
# arithmetic -- it cannot be MORE accurate than the deterministic rules
# layer already is for free, and measured accuracy here should be
# understood as validating that the injected signal is learnable, not as
# evidence this scorer is doing something rules can't. Its actual value-
# add over rules alone (if any) would be catching statistically-unusual-
# but-arithmetically-consistent patterns -- not exercised by this
# dataset's injection method, which only ever corrupts the stated total.

import pandas as pd

from core.model_store import load_or_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class PayrollAnomalyScorer:

    FRAUD_THRESHOLD = 0.7
    LEGIT_THRESHOLD = 0.3

    FEATURE_COLS = [
        "BasePay", "OvertimePay", "OtherPay", "Benefits",
        "Stated_TotalPay", "Stated_TotalPayBenefits",
    ]
    REQUIRED_COLS = ["BasePay", "OvertimePay", "OtherPay", "Stated_TotalPay"]

    def __init__(self, data_path="domains/payroll_hr/payroll/data/tier_ml_train.csv"):
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
                print(f"  [PayrollAnomalyScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [PayrollAnomalyScorer] No usable feature columns found. Scorer disabled.")
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
            print(f"  [PayrollAnomalyScorer] Trained on {len(df):,} rows "
                  f"({y.sum()} flagged, {y.mean():.1%} base rate)")
        except FileNotFoundError:
            print(f"  [PayrollAnomalyScorer] {data_path} not found. "
                  f"Run domains/payroll_hr/payroll/data/prepare_tier_ml_data.py first.")
        except Exception as e:
            print(f"  [PayrollAnomalyScorer] Training failed: {e}")

    def score(self, record: dict) -> float:
        if not self.trained:
            return 0.5
        try:
            row = {col: record.get(col, 0) for col in self.feature_cols}
            X = pd.DataFrame([row])
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [PayrollAnomalyScorer] Scoring error: {e}")
            return 0.5

    def score_batch(self, df: pd.DataFrame):
        import numpy as np

        if not self.trained:
            return np.full(len(df), 0.5)
        X = df.reindex(columns=self.feature_cols).fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]
