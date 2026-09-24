# domains/banking/fraud/tools/generalizable_scorer.py
# Tier 2 scorer -- for uploads that don't carry creditcard.csv's V1-V28
# schema (see detector_agent.py's module docstring for the full Tier 1/2/3
# picture). This is NOT "MLScorer with fewer columns": that was tried first
# (Amount+hour only, still from creditcard.csv) and empirically failed --
# ROC-AUC 0.66, precision 0.3% on the real holdout, because this dataset's
# only non-PCA columns (Amount, Time) carry almost no signal on their own
# and there's no account/merchant identifier to engineer anything richer
# from. Verified before rejecting it, not assumed.
#
# Instead this trains on a SECOND, real-world-shaped dataset:
# https://www.kaggle.com/datasets/mdtalhask/ai-powered-banking-fraud-detection-dataset-2025
# (Banking_Fraud_Dataset.csv via prepare_tier2_data.py) -- Transaction_Type,
# Account_Age, Credit_Score, Previous_Fraud, Device_Used, hour. Measured
# AUC 0.739 / F1 0.516 on its own holdout (with Previous_Fraud; 0.589/0.277
# without -- Previous_Fraud is legitimate repeat-offender signal, not
# leakage, but it's most of this scorer's real power).
#
# Two honest caveats this domain inherits from that dataset, unlike Tier 1:
#   - Only 10,000 rows (vs. Tier 1's ~285K) -- less data to learn from.
#   - 28% fraud rate, vastly higher than any real-world base rate (or
#     Tier 1's real 0.13%) -- this is a synthetic teaching dataset, not
#     real bank data. FRAUD_THRESHOLD/LEGIT_THRESHOLD below are calibrated
#     against ITS OWN base rate and are NOT the same thresholds Tier 1
#     uses -- do not assume they transfer.
#
# Amount is treated as USD here (no currency column, but Credit_Score's
# 300-850 range is the US FICO scale) -- consistent with Tier 1's
# EUR->USD conversion in prepare_data.py, so both tiers and
# rule_based_filter()'s USD-labeled thresholds agree on currency.

import pandas as pd

from core.model_store import load_or_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class GeneralizableScorer:

    # Calibrated against this dataset's own ~28% base rate via a holdout
    # sweep -- deliberately NOT MLScorer's 0.7/0.3 (those were tuned for
    # Tier 1's ~0.13% base rate and would resolve almost nothing here).
    FRAUD_THRESHOLD = 0.6
    LEGIT_THRESHOLD = 0.2

    FEATURE_COLS = [
        "Transaction_Amount", "Transaction_Type", "Account_Age",
        "Credit_Score", "Previous_Fraud", "Device_Used", "hour",
    ]
    CATEGORICAL_COLS = ["Transaction_Type", "Device_Used"]

    # The columns detect_tier() in detector_agent.py checks for to decide
    # whether an upload matches THIS schema well enough to route here.
    REQUIRED_COLS = ["Transaction_Amount", "Account_Age", "Credit_Score"]

    def __init__(self, data_path="domains/banking/fraud/data/tier2_train.csv"):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {c: LabelEncoder() for c in self.CATEGORICAL_COLS}
        self.trained = False
        load_or_train(self, data_path)

    def _train(self, data_path):
        try:
            df = pd.read_csv(data_path)
            available = [c for c in self.FEATURE_COLS if c in df.columns]
            missing = [c for c in self.FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"  [GeneralizableScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [GeneralizableScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].copy()
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            X = X.fillna(0)
            y = df["Is_Fraud"]

            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [GeneralizableScorer] Trained on {len(df):,} txns "
                  f"({y.sum()} fraud, {y.mean():.2%} base rate)")
        except FileNotFoundError:
            print(f"  [GeneralizableScorer] {data_path} not found. "
                  f"Run domains/banking/fraud/data/prepare_tier2_data.py first.")
        except Exception as e:
            print(f"  [GeneralizableScorer] Training failed: {e}")

    def score(self, transaction: dict) -> float:
        """Returns fraud probability 0.0-1.0. Returns 0.5 (borderline) if
        untrained -- same fail-safe convention as MLScorer."""
        if not self.trained:
            return 0.5
        try:
            row = {}
            for col in self.feature_cols:
                val = transaction.get(col, 0)
                if col in self.CATEGORICAL_COLS:
                    encoder = self.encoders[col]
                    val = str(val)
                    if val not in encoder.classes_:
                        # Unseen category (e.g. a device type this training
                        # data never had) -- fall back to the most common
                        # class rather than crash.
                        val = encoder.classes_[0]
                    val = encoder.transform([val])[0]
                row[col] = val
            X = pd.DataFrame([row])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [GeneralizableScorer] Scoring error: {e}")
            return 0.5

    def score_batch(self, df: pd.DataFrame):
        """Vectorised scoring, mirroring MLScorer.score_batch -- used by
        eval_harness-style measurement, not by the live per-transaction path."""
        import numpy as np

        if not self.trained:
            return np.full(len(df), 0.5)
        X = df[self.feature_cols].copy()
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                encoder = self.encoders[col]
                known = set(encoder.classes_)
                X[col] = X[col].astype(str).apply(lambda v: v if v in known else encoder.classes_[0])
                X[col] = encoder.transform(X[col])
        X = X.fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]
