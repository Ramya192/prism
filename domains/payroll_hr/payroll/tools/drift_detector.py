# domains/payroll_hr/payroll/tools/drift_detector.py
# Isolation Forest -- unsupervised companion to PayrollAnomalyScorer
# (Tier 1), same purpose as domains/banking/fraud/tools/drift_detector.py:
# answers "does this record look statistically unlike training data,"
# not "is this fraud." See that file's module docstring for the full
# rationale (fills the "drift detection" gap from the original planning
# doc) and detector_agent.py-style wiring pattern if/when this gets
# plugged into the live decision flow.

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from domains.payroll_hr.payroll.tools.anomaly_scorer import PayrollAnomalyScorer


class DriftDetector:
    def __init__(self, data_path="domains/payroll_hr/payroll/data/tier_ml_train.csv", contamination="auto"):
        self.scaler = StandardScaler()
        self.model = None
        self.trained = False
        self._train(data_path, contamination)

    def _train(self, data_path, contamination):
        try:
            df = pd.read_csv(data_path)
            available = [c for c in PayrollAnomalyScorer.FEATURE_COLS if c in df.columns]
            if not available:
                print("  [DriftDetector] No usable feature columns found. Disabled.")
                return

            X = df[available].fillna(0)
            X_scaled = self.scaler.fit_transform(X)

            self.model = IsolationForest(
                n_estimators=150, contamination=contamination, random_state=42, n_jobs=-1,
            )
            self.model.fit(X_scaled)
            self.feature_cols = available
            self.trained = True
            print(f"  [DriftDetector] Trained (unsupervised) on {len(df):,} rows, "
                  f"{len(available)} features -- no label used.")
        except FileNotFoundError:
            print(f"  [DriftDetector] {data_path} not found.")
        except Exception as e:
            print(f"  [DriftDetector] Training failed: {e}")

    def is_outlier(self, record: dict) -> bool:
        if not self.trained:
            return False
        row = {col: record.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([row])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return bool(self.model.predict(X_scaled)[0] == -1)

    def anomaly_score(self, record: dict) -> float:
        if not self.trained:
            return 0.0
        row = {col: record.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([row])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return float(self.model.decision_function(X_scaled)[0])

    def score_batch(self, df: pd.DataFrame):
        import numpy as np

        if not self.trained:
            return np.zeros(len(df))
        X = df.reindex(columns=self.feature_cols).fillna(0)
        return self.model.decision_function(self.scaler.transform(X))
