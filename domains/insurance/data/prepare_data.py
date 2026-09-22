# domains/insurance/data/prepare_data.py
# Rebuilds the train/holdout split for healthcare's claim-level fraud
# detector.
#
# Source: https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset
# (10,000 claims, real per-CLAIM Is_Fraud label, 8.29% fraud rate, USD).
#
# This is the SECOND dataset considered for this domain, not the first.
# The original plan (per configs/healthcare.yaml's original stub comment)
# pointed at rohitrox/healthcare-provider-fraud-detection-analysis --
# real Medicare data, but its fraud label only exists at the PROVIDER
# level (aggregated across all of a provider's claims), which conflicts
# with how every other domain in this platform works: bfsi_fraud judges
# the transaction, payroll judges the payslip/row -- never the entity
# that produced it. Scoring per-provider here would have meant either
# breaking that pattern, or inheriting the provider's label onto every
# individual claim as a noisy proxy (statistically shaky -- a claim from
# a flagged provider isn't necessarily itself fraudulent). This dataset
# has a genuine claim-level label instead, avoiding that compromise
# entirely -- verified learnable before committing to it (AUC 0.9915,
# F1 0.8403 on a held-out random split), not assumed.
#
# No currency field, but Patient_State values (NY, etc.) and Insurance_Type
# (Medicaid) are US-specific -- USD, consistent with bfsi_fraud Tier 1 and
# payroll Tier 1's currency (see docs/DATA_CONVENTIONS.md).
#
# Usage (from the prism/ repo root):
#   python domains/insurance/data/prepare_data.py
# Requires: domains/insurance/data/raw/healthcare_fraud_detection.csv
# (not committed -- download via `kaggle datasets download -d
# nudratabbas/healthcare-fraud-detection-dataset`, unzip into that folder).

import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
RAW_PATH = _HERE / "raw" / "healthcare_fraud_detection.csv"
TRAIN_PATH = _HERE / "train.csv"
HOLDOUT_PATH = _HERE / "test_holdout.csv"
SEED = 42

print("Loading healthcare claims raw data ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")

# A handful of rows are missing Insurance_Type/Provider_Specialty/
# Prior_Visits_12m (~3.5%) -- drop rather than impute, small enough not
# to matter and avoids inventing values for a fraud-relevant field.
before = len(df)
df = df.dropna(subset=["Insurance_Type", "Provider_Specialty", "Prior_Visits_12m"]).reset_index(drop=True)
print(f"  Dropped {before - len(df)} rows with missing categorical/count fields")

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
cut = int(len(df) * 0.70)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

print(f"\n  Train: {len(train_df):,} rows, {train_df.Is_Fraud.mean():.2%} fraud rate")
print(f"  Test:  {len(test_df):,} rows, {test_df.Is_Fraud.mean():.2%} fraud rate")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")
