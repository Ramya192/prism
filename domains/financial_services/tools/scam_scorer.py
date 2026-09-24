# domains/financial_services/tools/scam_scorer.py
# Tier 1 scorer -- per-blockchain-transaction scam detection. See
# data/prepare_data.py's module docstring for the dataset search (two
# other real candidates rejected first) and the original Central Asia
# Fintech wallet dataset this replaced as Tier 1 (benched, not deleted --
# domains/financial_services/data/backup_wallet_raw/).
#
# RandomForest, class_weight="balanced" -- no gradient-boosting swap
# attempted; this dataset's signal, while real, is still diffuse enough
# (top feature 34% importance, no single dominant giveaway) that a
# boosting swap is unlikely to move the needle meaningfully, matching
# the same reasoning that skipped a swap for the wallet dataset this
# replaced.
#
# Measured on a real train/holdout split (test_size=0.2, random_state=42):
#   AUC 0.908, F1 0.425, Precision 0.292, Recall 0.776.
# Top features: is_cross_chain (0.34 -- cross-chain bridge transfers are
# a well-known real laundering vector, obscuring fund trails; measured
# 23.7% scam rate when true vs. 3.6% when false -- a real, plausible
# signal, not a giveaway), sender_wallet_age_days (0.26 -- newer wallets
# skew scammier: median 226.5 days for scam vs. 427 for legit),
# failed_txn_ratio_sender (0.17). velocity_score and anomaly_score look
# pre-computed by name but measured weak (correlation 0.077 and 0.008
# respectively) -- checked individually before trusting them, not
# assumed safe or assumed leaky either way.

import pandas as pd

from core.model_store import load_or_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ScamScorer:

    # Calibrated against this dataset's own ~7.25% base rate via a
    # holdout sweep -- not the same thresholds as Tier 2's.
    FRAUD_THRESHOLD = 0.6
    LEGIT_THRESHOLD = 0.2

    NUMERIC_COLS = [
        "sender_wallet_age_days", "receiver_wallet_age_days", "transaction_amount_usd",
        "gas_fee_usd", "num_prev_transactions_sender", "num_prev_transactions_receiver",
        "avg_txn_interval_sender_min", "is_cross_chain", "failed_txn_ratio_sender",
        "velocity_score", "anomaly_score",
    ]
    CATEGORICAL_COLS = ["blockchain", "transaction_type", "token_type", "platform"]
    FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

    # What detect_tier() checks for to decide an upload matches this schema.
    REQUIRED_COLS = ["transaction_amount_usd", "sender_wallet_age_days", "is_cross_chain"]

    def __init__(self, data_path="domains/financial_services/data/train.csv"):
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
                print(f"  [ScamScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [ScamScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].copy()
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].fillna("missing").astype(str))
            X = X.fillna(0)
            y = df["is_scam"]

            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [ScamScorer] Trained on {len(df):,} txns "
                  f"({y.sum()} scam, {y.mean():.2%} base rate)")
        except FileNotFoundError:
            print(f"  [ScamScorer] {data_path} not found. "
                  f"Run domains/financial_services/data/prepare_data.py first.")
        except Exception as e:
            print(f"  [ScamScorer] Training failed: {e}")

    def score(self, record: dict) -> float:
        """Returns scam probability 0.0-1.0. Returns 0.5 (borderline) if
        untrained -- same fail-safe convention as every other scorer."""
        if not self.trained:
            return 0.5
        try:
            row = {}
            for col in self.feature_cols:
                val = record.get(col, 0)
                if col in self.CATEGORICAL_COLS:
                    encoder = self.encoders[col]
                    val = str(val)
                    if val not in encoder.classes_:
                        val = encoder.classes_[0]
                    val = encoder.transform([val])[0]
                row[col] = val
            X = pd.DataFrame([row])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [ScamScorer] Scoring error: {e}")
            return 0.5

    def score_batch(self, df: pd.DataFrame):
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
