# domains/insurance/tools/generalizable_scorer.py
# Tier 2 scorer for healthcare's claim-level detector -- a SECOND,
# genuinely differently-shaped claims schema than Tier 1's (nudratabbas'
# Claim_Amount/Approved_Amount/Days_Between_Service_and_Claim):
# tejalaveti2306's Policy_Number/Claim_Date/Provider_Patient_Distance_Miles/
# Number_of_Previous_Claims_* convention. See data/prepare_tier2_data.py's
# module docstring for the full search story (two other candidates
# checked and rejected first) and the honest caveat this dataset carries.
#
# Measured AUC 0.857, F1 0.667 on a random holdout -- genuinely learnable,
# signal spread across 5+ features (Number_of_Procedures, Claim_Amount,
# Provider_Type, Provider_Patient_Distance_Miles, Admission_Type all
# contribute, no single dominant giveaway).
#
# HONEST CAVEAT: this dataset's Claim_Amount values (mean ~$1.24M) are
# roughly 200-1000x larger than Tier 1's real-claim-shaped amounts
# (~$570 mean) -- not a currency issue (both are already USD), just an
# implausible numeric range in this synthetic generator. Kept and
# documented rather than hidden, same treatment as banking fraud Tier 2's
# unrealistic 28% base rate.

import pandas as pd

from core.model_store import load_or_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ClaimGeneralizableScorer:

    # Calibrated against this dataset's own ~25% base rate -- not the
    # same thresholds as Tier 1's ClaimFraudScorer (different dataset,
    # different score distribution).
    FRAUD_THRESHOLD = 0.6
    LEGIT_THRESHOLD = 0.25

    NUMERIC_COLS = [
        "Claim_Amount", "Patient_Age", "Number_of_Procedures", "Length_of_Stay_Days",
        "Deductible_Amount", "CoPay_Amount", "Number_of_Previous_Claims_Patient",
        "Number_of_Previous_Claims_Provider", "Provider_Patient_Distance_Miles",
    ]
    CATEGORICAL_COLS = [
        "Patient_Gender", "Provider_Type", "Provider_Specialty",
        "Admission_Type", "Discharge_Type", "Service_Type", "Claim_Submitted_Late",
    ]
    FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
    REQUIRED_COLS = ["Claim_Amount", "Policy_Number", "Provider_Patient_Distance_Miles"]

    def __init__(self, data_path="domains/insurance/data/tier2_train.csv"):
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
                print(f"  [ClaimGeneralizableScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [ClaimGeneralizableScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].copy()
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            X = X.fillna(0)
            y = df["Is_Fraudulent"]

            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [ClaimGeneralizableScorer] Trained on {len(df):,} claims "
                  f"({y.sum()} fraud, {y.mean():.2%} base rate)")
        except FileNotFoundError:
            print(f"  [ClaimGeneralizableScorer] {data_path} not found. "
                  f"Run domains/insurance/data/prepare_tier2_data.py first.")
        except Exception as e:
            print(f"  [ClaimGeneralizableScorer] Training failed: {e}")

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
                        val = encoder.classes_[0]
                    val = encoder.transform([val])[0]
                row[col] = val
            X = pd.DataFrame([row])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [ClaimGeneralizableScorer] Scoring error: {e}")
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
