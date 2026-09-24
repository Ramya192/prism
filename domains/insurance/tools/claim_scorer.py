# domains/insurance/tools/claim_scorer.py
# ML fraud scorer for healthcare's claim-level detector -- see
# agents/healthcare_detector_agent.py's module docstring for how this
# fits alongside the rules layer and LLM fallback, and
# data/prepare_data.py's module docstring for why this claim-level
# dataset was chosen over the originally-planned provider-level one.
#
# Trained on real per-claim fraud labels (not synthetic/injected, unlike
# every scorer built so far in this repo). Top feature by a wide margin:
# Days_Between_Service_and_Claim (56% importance in the original
# RandomForest) -- fraudulent claims are submitted much faster than
# legitimate ones (mean 2.97 days vs. 15.45 days in the training data),
# a real, documented healthcare-fraud pattern (rushed/phantom billing),
# not an artifact of label construction.
#
# LightGBM, not RandomForest: swapped after benchmarking RandomForest vs.
# XGBoost vs. LightGBM on this identical train/test split -- RF scored
# AUC 0.994/F1 0.869, LightGBM AUC 0.998/F1 0.932 (XGBoost was
# comparable, AUC 0.998/F1 0.921). A real, measured gain, not a default
# swap -- this is the only one of this domain's tiers that changed;
# ClaimGeneralizableScorer (Tier 2) stayed on RandomForest since the
# benchmark showed no clear win there.

import pandas as pd

from core.model_store import load_or_train
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ClaimFraudScorer:

    # Re-calibrated for LightGBM's score distribution via a fresh holdout
    # sweep after the RandomForest->LightGBM swap (0.6/0.2 was tuned for
    # RandomForest's output; LightGBM's is different enough to be worth
    # re-checking, not assumed to transfer) -- not the same thresholds as
    # any other domain's scorer.
    FRAUD_THRESHOLD = 0.7
    LEGIT_THRESHOLD = 0.3

    NUMERIC_COLS = [
        "Patient_Age", "Claim_Amount", "Approved_Amount", "Days_Between_Service_and_Claim",
        "Number_of_Claims_Per_Provider_Monthly", "Length_of_Stay", "Chronic_Condition_Flag",
        "Prior_Visits_12m",
    ]
    CATEGORICAL_COLS = ["Patient_Gender", "Insurance_Type", "Provider_Specialty", "Visit_Type"]
    FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
    REQUIRED_COLS = ["Claim_Amount", "Approved_Amount", "Days_Between_Service_and_Claim"]

    def __init__(self, data_path="domains/insurance/data/train.csv"):
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
                print(f"  [ClaimFraudScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [ClaimFraudScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].copy()
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            X = X.fillna(0)
            y = df["Is_Fraud"]

            X_scaled = self.scaler.fit_transform(X)
            self.model = LGBMClassifier(
                n_estimators=150, max_depth=6, is_unbalance=True,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [ClaimFraudScorer] Trained on {len(df):,} claims "
                  f"({y.sum()} fraud, {y.mean():.2%} base rate)")
        except FileNotFoundError:
            print(f"  [ClaimFraudScorer] {data_path} not found. "
                  f"Run domains/insurance/data/prepare_data.py first.")
        except Exception as e:
            print(f"  [ClaimFraudScorer] Training failed: {e}")

    def score(self, claim: dict) -> float:
        if not self.trained:
            return 0.5
        try:
            row = {}
            for col in self.feature_cols:
                val = claim.get(col, 0)
                if col in self.CATEGORICAL_COLS:
                    encoder = self.encoders[col]
                    val = str(val)
                    if val not in encoder.classes_:
                        val = encoder.classes_[0]  # unseen category -- fall back, don't crash
                    val = encoder.transform([val])[0]
                row[col] = val
            X = pd.DataFrame([row])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [ClaimFraudScorer] Scoring error: {e}")
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
