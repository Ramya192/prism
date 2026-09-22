# domains/insurance/tools/drift_detector.py
# Isolation Forest -- unsupervised companion to ClaimFraudScorer (Tier 1),
# same purpose as domains/banking/fraud/tools/drift_detector.py: answers
# "does this claim look statistically unlike training data," not "is
# this fraud." See that file's module docstring for the full rationale.

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

from domains.insurance.tools.claim_scorer import ClaimFraudScorer


class DriftDetector:
    def __init__(self, data_path="domains/insurance/data/train.csv", contamination="auto"):
        self.scaler = StandardScaler()
        self.encoders = {c: LabelEncoder() for c in ClaimFraudScorer.CATEGORICAL_COLS}
        self.model = None
        self.trained = False
        self._train(data_path, contamination)

    def _train(self, data_path, contamination):
        try:
            df = pd.read_csv(data_path)
            available = [c for c in ClaimFraudScorer.FEATURE_COLS if c in df.columns]
            if not available:
                print("  [DriftDetector] No usable feature columns found. Disabled.")
                return

            X = df[available].copy()
            for col in ClaimFraudScorer.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].fillna("missing").astype(str))
            X = X.fillna(0)
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

    def _encode(self, record: dict) -> dict:
        row = {}
        for col in self.feature_cols:
            val = record.get(col, 0)
            if col in ClaimFraudScorer.CATEGORICAL_COLS:
                encoder = self.encoders[col]
                val = str(val)
                if val not in encoder.classes_:
                    val = encoder.classes_[0]
                val = encoder.transform([val])[0]
            row[col] = val
        return row

    def is_outlier(self, record: dict) -> bool:
        if not self.trained:
            return False
        X = pd.DataFrame([self._encode(record)])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return bool(self.model.predict(X_scaled)[0] == -1)

    def anomaly_score(self, record: dict) -> float:
        if not self.trained:
            return 0.0
        X = pd.DataFrame([self._encode(record)])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return float(self.model.decision_function(X_scaled)[0])

    def score_batch(self, df: pd.DataFrame):
        import numpy as np

        if not self.trained:
            return np.zeros(len(df))
        X = df[self.feature_cols].copy()
        for col in ClaimFraudScorer.CATEGORICAL_COLS:
            if col in X.columns:
                encoder = self.encoders[col]
                known = set(encoder.classes_)
                X[col] = X[col].fillna("missing").astype(str).apply(lambda v: v if v in known else encoder.classes_[0])
                X[col] = encoder.transform(X[col])
        X = X.fillna(0)
        return self.model.decision_function(self.scaler.transform(X))
