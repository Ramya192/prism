# domains/bfsi_fraud/tools/ml_scorer.py
# Ported from fraud_detection/tools/ml_scorer.py — logic unchanged.
# Random Forest scorer trained on all 30 features. Caller picks the data file
# (see class docstring below) — FraudDetectorAgent trains it on
# domains/bfsi_fraud/data/transactions_balanced.csv by default; eval_harness.py
# points it at data/train.csv instead so evaluation never touches the holdout.
# Used by FraudDetectorAgent as the first decision layer (after rule-based filter)
# Returns a fraud probability score: 0.0 (definitely legit) -> 1.0 (definitely fraud)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLScorer:

    # Thresholds for fast-path decisions (skip LLM)
    FRAUD_THRESHOLD = 0.7    # score >= 0.7 → HIGH RISK, no LLM needed
    LEGIT_THRESHOLD = 0.3    # score <= 0.3 → LOW RISK,  no LLM needed
    # score between 0.3 and 0.7 → BORDERLINE, send to LLM

    # All features the model trains on
    FEATURE_COLS = (
        [f"V{i}" for i in range(1, 29)]   # V1 through V28
        + ["Amount", "Time", "hour"]
    )

    def __init__(self, data_path="domains/bfsi_fraud/data/train.csv"):
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._train(data_path)

    # ── Training ────────────────────────────────────────────────────────
    def _train(self, data_path):
        try:
            df = pd.read_csv(data_path)

            # Identify which feature columns actually exist in this CSV
            available = [c for c in self.FEATURE_COLS if c in df.columns]
            missing   = [c for c in self.FEATURE_COLS if c not in df.columns]

            if missing:
                print(f"  [MLScorer] Missing columns (will be ignored): {missing}")

            if not available:
                print("  [MLScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].fillna(0)
            y = df["is_Fraud"]
            X_scaled = self.scaler.fit_transform(X)

            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced",     # the ONLY imbalance mechanism now
                random_state=42, n_jobs=-1
            )
            self.model.fit(X_scaled, y)      # fit on the full training window

            self.feature_cols = available
            self.trained = True
            print(f"  [MLScorer] Trained on {len(df):,} txns "
                  f"({y.sum()} fraud, {y.mean():.4%} base rate)")

        except FileNotFoundError:
            print(f"  [MLScorer] {data_path} not found. "
                  f"Run domains/bfsi_fraud/data/prepare_data.py first.")
        except Exception as e:
            print(f"  [MLScorer] Training failed: {e}")

    # ── Scoring ─────────────────────────────────────────────────────────
    def score(self, transaction: dict) -> float:
        """
        Returns fraud probability 0.0-1.0 for a single transaction dict.
        Returns 0.5 (borderline) if model is not trained.
        """
        if not self.trained:
            return 0.5

        try:
            # Build feature row — use 0 for any missing column
            row = {col: transaction.get(col, 0) for col in self.feature_cols}
            X = pd.DataFrame([row])
            X_scaled = self.scaler.transform(X)
            prob = self.model.predict_proba(X_scaled)[0][1]   # P(fraud)
            return float(prob)

        except Exception as e:
            print(f"  [MLScorer] Scoring error: {e}")
            return 0.5   # default to borderline — let LLM decide

    def score_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring. One predict_proba call for the whole frame."""
        if not self.trained:
            return np.full(len(df), 0.5)
        X = df.reindex(columns=self.feature_cols).fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def label(self, score: float) -> str:
        """Converts score to a human-readable label."""
        if score >= self.FRAUD_THRESHOLD:
            return "HIGH"
        elif score <= self.LEGIT_THRESHOLD:
            return "LOW"
        else:
            return "MEDIUM (borderline)"
