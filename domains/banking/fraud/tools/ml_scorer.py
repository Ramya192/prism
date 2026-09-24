# domains/banking/fraud/tools/ml_scorer.py
# Ported from fraud_detection/tools/ml_scorer.py — logic unchanged apart
# from the feature_cols constructor param (currently unused by anything
# else in the codebase, kept because MLScorer's own eval/calibration
# scripts (evaluate.py) may still want to try feature subsets of THIS
# dataset). This is Tier 1 in detector_agent.py's Tier1/Tier2/Tier3
# routing — the full-accuracy scorer for uploads that carry the exact
# V1-V28 schema this dataset was built with.
#
# An earlier attempt used this same class with a reduced feature_cols
# (Amount+hour only, dropping V1-V28) as a "Tier 2" for uploads without
# those PCA columns. Measured and rejected: AUC 0.66, precision 0.3% on
# the real holdout — this dataset's only non-PCA columns (Amount, Time)
# carry almost no signal on their own, and there's no account/merchant
# identifier here to engineer anything richer from. Tier 2 is now
# generalizable_scorer.py, trained on a second, real-world-shaped dataset
# instead — see that file and detector_agent.py's module docstring.
#
# Random Forest scorer. Caller picks the data file (see class docstring
# below) — FraudDetectorAgent and eval_harness.py both use data/train.csv, so
# the deployed model is the one the published eval numbers measure (and
# evaluation never touches the holdout).
# Used by FraudDetectorAgent as the first decision layer (after rule-based filter)
# Returns a fraud probability score: 0.0 (definitely legit) -> 1.0 (definitely fraud)

import pandas as pd

from core.model_store import load_or_train
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLScorer:

    # Thresholds for fast-path decisions (skip LLM)
    FRAUD_THRESHOLD = 0.7    # score >= 0.7 → HIGH RISK, no LLM needed
    LEGIT_THRESHOLD = 0.3    # score <= 0.3 → LOW RISK,  no LLM needed
    # score between 0.3 and 0.7 → BORDERLINE, send to LLM

    # The full feature set this dataset was built for — best accuracy, but
    # V1-V28 are anonymized PCA components specific to *this* Kaggle
    # dataset, so this schema only matches an upload that is itself
    # Kaggle-shaped (e.g. the sample file offered in the UI).
    FEATURE_COLS = (
        [f"V{i}" for i in range(1, 29)]   # V1 through V28
        + ["Amount", "Time", "hour"]
    )

    def __init__(self, data_path="domains/banking/fraud/data/train.csv", feature_cols=None):
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        self._requested_cols = feature_cols or self.FEATURE_COLS
        load_or_train(self, data_path, signature=repr(self._requested_cols))

    # ── Training ────────────────────────────────────────────────────────
    def _train(self, data_path):
        try:
            df = pd.read_csv(data_path)

            # Identify which feature columns actually exist in this CSV
            available = [c for c in self._requested_cols if c in df.columns]
            missing   = [c for c in self._requested_cols if c not in df.columns]

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
                  f"Run domains/banking/fraud/data/prepare_data.py first.")
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
