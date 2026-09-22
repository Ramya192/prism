# domains/banking/fraud/tools/drift_detector.py
# Isolation Forest -- an UNSUPERVISED companion to MLScorer (the
# supervised Tier 1 classifier), answering a different question:
# "does this transaction look like anything in the training
# distribution at all," not "is this transaction fraud."
#
# Why this exists: MLScorer can only recognize fraud patterns that
# resemble what was in its labeled training data. A transaction shaped
# nothing like anything seen before (a genuinely novel pattern, or the
# population has drifted since training) can still get a confident
# supervised score without ever signaling "this is unlike anything I've
# seen." This fills that gap -- fills the "Model & data drift detection"
# item the original Aug-14 planning doc flagged as unaddressed.
#
# How it works: builds a forest of random trees that isolate points via
# random feature splits. Outliers get isolated in very few splits (short
# average path length across the forest); typical points, surrounded by
# many similar points, take many splits. No label is used at all --
# trained on the same training data as MLScorer, but blind to is_Fraud.
#
# This is a companion signal, not a replacement -- see how it's used in
# detector_agent.py (or left as a standalone tool if not yet wired in).

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from domains.banking.fraud.tools.ml_scorer import MLScorer


class DriftDetector:
    def __init__(self, data_path="domains/banking/fraud/data/train.csv", contamination="auto"):
        self.scaler = StandardScaler()
        self.model = None
        self.trained = False
        self._train(data_path, contamination)

    def _train(self, data_path, contamination):
        try:
            df = pd.read_csv(data_path)
            available = [c for c in MLScorer.FEATURE_COLS if c in df.columns]
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

    def is_outlier(self, transaction: dict) -> bool:
        """True if this transaction looks statistically unlike the
        training distribution -- a caveat signal, not a fraud verdict."""
        if not self.trained:
            return False
        row = {col: transaction.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([row])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return bool(self.model.predict(X_scaled)[0] == -1)  # -1 = outlier, 1 = inlier

    def anomaly_score(self, transaction: dict) -> float:
        """Raw decision_function score -- more negative = more anomalous.
        Exposed for calibration/inspection; is_outlier() is the normal
        entry point."""
        if not self.trained:
            return 0.0
        row = {col: transaction.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([row])[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return float(self.model.decision_function(X_scaled)[0])

    def score_batch(self, df: pd.DataFrame):
        import numpy as np

        if not self.trained:
            return np.zeros(len(df))
        X = df.reindex(columns=self.feature_cols).fillna(0)
        return self.model.decision_function(self.scaler.transform(X))
