# domains/payroll_hr/hr/tools/posting_scorer.py
# ML fraud scorer for hr's job-posting detector -- see
# agents/hr_detector_agent.py's module docstring for how this fits
# alongside the rules layer and LLM fallback, and data/prepare_data.py's
# module docstring for the dataset choice and why salary amounts are
# deliberately excluded (multi-currency risk on a mostly-missing field).
#
# Trained on real per-posting fraud labels (EMSCAD, University of the
# Aegean -- an actual academic dataset, not synthetic). Top features (in
# the original RandomForest): has_company_profile (19%), has_company_logo
# (12%), description_length (11%), country_code, industry, function --
# no single dominant giveaway, and all are structural/presence signals
# rather than raw currency amounts.
#
# XGBoost, not RandomForest: swapped after benchmarking RandomForest vs.
# XGBoost vs. LightGBM on this identical train/test split -- RF scored
# AUC 0.969/F1 0.666, XGBoost AUC 0.966/F1 0.716 (LightGBM was close to
# RF, AUC 0.967/F1 0.662). XGBoost's raw AUC is very slightly lower, but
# its precision is meaningfully better (0.703 vs. 0.591 at the same 0.5
# cutoff), which is what actually drives the F1 gain -- a real, measured
# tradeoff, not a default swap.

import numpy as np
import pandas as pd

from core.model_store import load_or_train
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PostingFraudScorer:

    # Calibrated against this dataset's own ~4.8% base rate via a holdout
    # sweep -- re-verified (unchanged) after the RandomForest->XGBoost
    # swap; not the same thresholds as any other domain's scorer.
    FRAUD_THRESHOLD = 0.6
    LEGIT_THRESHOLD = 0.2

    NUMERIC_COLS = [
        "telecommuting", "has_company_logo", "has_questions", "has_salary_range",
        "has_company_profile", "has_department", "has_requirements", "has_benefits",
        "description_length",
    ]
    CATEGORICAL_COLS = [
        "employment_type", "required_experience", "required_education",
        "industry", "function", "country_code",
    ]
    FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
    REQUIRED_COLS = ["title", "description", "telecommuting", "has_company_logo", "has_questions"]

    def __init__(self, data_path="domains/payroll_hr/hr/data/train.csv"):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {c: LabelEncoder() for c in self.CATEGORICAL_COLS}
        self.trained = False
        load_or_train(self, data_path)

    @staticmethod
    def _derive_features(record: dict) -> dict:
        """Computes the same structural/presence features prepare_data.py
        derives from the raw CSV, from a single record dict at inference
        time -- so callers can pass raw-ish fields (title, description,
        salary_range, etc.) without pre-computing has_salary_range etc.
        themselves. If the derived fields are already present (e.g. when
        scoring a row read back from train.csv), those are left as-is."""
        derived = dict(record)
        if "has_salary_range" not in derived:
            derived["has_salary_range"] = int(bool(record.get("salary_range")))
        if "has_company_profile" not in derived:
            derived["has_company_profile"] = int(bool(record.get("company_profile")))
        if "has_department" not in derived:
            derived["has_department"] = int(bool(record.get("department")))
        if "has_requirements" not in derived:
            derived["has_requirements"] = int(bool(record.get("requirements")))
        if "has_benefits" not in derived:
            derived["has_benefits"] = int(bool(record.get("benefits")))
        if "description_length" not in derived:
            derived["description_length"] = len(record.get("description") or "")
        if "country_code" not in derived:
            location = record.get("location") or ""
            derived["country_code"] = location.split(",")[0].strip()
        return derived

    def _train(self, data_path):
        try:
            df = pd.read_csv(data_path)
            available = [c for c in self.FEATURE_COLS if c in df.columns]
            missing = [c for c in self.FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"  [PostingFraudScorer] Missing columns (will be ignored): {missing}")
            if not available:
                print("  [PostingFraudScorer] No usable feature columns found. Scorer disabled.")
                return

            X = df[available].copy()
            for col in self.CATEGORICAL_COLS:
                if col in X.columns:
                    X[col] = self.encoders[col].fit_transform(X[col].fillna("missing").astype(str))
            X = X.fillna(0)
            y = df["fraudulent"]

            X_scaled = self.scaler.fit_transform(X)
            # XGBoost has no class_weight="balanced" equivalent -- scale_pos_weight
            # is the direct translation (ratio of negative to positive examples).
            pos = int(y.sum())
            neg = int(len(y) - pos)
            scale_pos_weight = neg / max(pos, 1)
            self.model = XGBClassifier(
                n_estimators=150, max_depth=6, scale_pos_weight=scale_pos_weight,
                random_state=42, n_jobs=-1, eval_metric="logloss",
            )
            self.model.fit(X_scaled, y)
            self.feature_cols = available
            self.trained = True
            print(f"  [PostingFraudScorer] Trained on {len(df):,} postings "
                  f"({y.sum()} fraud, {y.mean():.2%} base rate)")
        except FileNotFoundError:
            print(f"  [PostingFraudScorer] {data_path} not found. "
                  f"Run domains/payroll_hr/hr/data/prepare_data.py first.")
        except Exception as e:
            print(f"  [PostingFraudScorer] Training failed: {e}")

    def score(self, record: dict) -> float:
        if not self.trained:
            return 0.5
        try:
            record = self._derive_features(record)
            row = {}
            for col in self.feature_cols:
                val = record.get(col, 0)
                if col in self.CATEGORICAL_COLS:
                    encoder = self.encoders[col]
                    val = str(val) if val else "missing"
                    if val not in encoder.classes_:
                        val = "missing" if "missing" in encoder.classes_ else encoder.classes_[0]
                    val = encoder.transform([val])[0]
                row[col] = val
            X = pd.DataFrame([row])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            print(f"  [PostingFraudScorer] Scoring error: {e}")
            return 0.5

    def score_batch(self, df: pd.DataFrame):
        if not self.trained:
            return np.full(len(df), 0.5)
        X = df[self.feature_cols].copy()
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                encoder = self.encoders[col]
                known = set(encoder.classes_)
                fallback = "missing" if "missing" in known else encoder.classes_[0]
                X[col] = X[col].fillna("missing").astype(str).apply(lambda v: v if v in known else fallback)
                X[col] = encoder.transform(X[col])
        X = X.fillna(0)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]
