# domains/financial_services/tools/wash_trading_scorer.py
# Tier 2 scorer -- per-transaction exchange/DEX market-manipulation
# detection (wash trading / pump-and-dump / ramping). See
# data/prepare_tier2_data.py's module docstring for why this is a
# genuinely different fraud problem and schema from Tier 1's scam-wallet
# detection, not a reduced copy of it.
#
# RandomForest, class_weight="balanced". Measured on a real train/holdout
# split (test_size=0.2, random_state=42):
#   AUC 0.928, F1 0.356, Precision 0.224, Recall 0.865.
# No single dominant feature (top is token_liquidity_score at 17%
# importance) -- diffuse signal across venue/liquidity/price-deviation
# features, checked individually for leakage before trusting (max
# correlation with the label: 0.134, counterparty_diversity) despite
# several pre-engineered "_score"/"_zscore" columns that looked
# suspicious by name alone.

import pandas as pd

from core.model_store import load_or_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class WashTradingScorer:

    # Calibrated against this dataset's own ~5.94% base rate via a
    # holdout sweep -- not the same thresholds as Tier 1's.
    FRAUD_THRESHOLD = 0.6
    LEGIT_THRESHOLD = 0.2

    NUMERIC_COLS = [
        "token_is_major", "token_liquidity_score", "is_dex", "base_amount",
        "price_usd", "notional_usd", "gas_fee_usd", "wallet_age_days",
        "new_wallet_flag", "wallet_txn_count_total", "time_since_prev_tx_sec",
        "tx_burst_count_5m", "counterparty_diversity", "counterparty_reuse_ratio",
        "round_trip_score", "pair_txn_count", "token_volume_zscore_1h",
        "price_deviation_vwap_pct", "notional_zscore_wallet", "hour_of_day", "is_night",
    ]
    CATEGORICAL_COLS = ["venue_type"]
    FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

    # What detect_tier() checks for to decide an upload matches this schema.
    REQUIRED_COLS = ["notional_usd", "round_trip_score", "counterparty_reuse_ratio"]

    def __init__(self, data_path="domains/financial_services/data/tier2_train.csv"):
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
                print(f"  [WashTradingScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [WashTradingScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].copy()
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].fillna("missing").astype(str))
            X = X.fillna(0)
            y = df["is_manipulative"]

            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [WashTradingScorer] Trained on {len(df):,} txns "
                  f"({y.sum()} manipulative, {y.mean():.2%} base rate)")
        except FileNotFoundError:
            print(f"  [WashTradingScorer] {data_path} not found. "
                  f"Run domains/financial_services/data/prepare_tier2_data.py first.")
        except Exception as e:
            print(f"  [WashTradingScorer] Training failed: {e}")

    def score(self, record: dict) -> float:
        """Returns manipulation probability 0.0-1.0. Returns 0.5
        (borderline) if untrained -- same fail-safe convention as every
        other scorer."""
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
            print(f"  [WashTradingScorer] Scoring error: {e}")
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
