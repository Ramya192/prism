# domains/banking/fraud/data/prepare_tier2_data.py
# Companion to prepare_data.py, for the SECOND dataset this domain trains
# on: Tier 2's generalizable scorer needs real-world-shaped columns
# (merchant/device/credit-history-style fields), which creditcard.csv
# structurally cannot provide -- see detector_agent.py's module docstring
# for why Tier 1/Tier 2 exist and generalizable_scorer.py's docstring for
# why a reduced-feature version of creditcard.csv was tried and rejected
# (empirically: ROC-AUC 0.66, precision 0.3% on the real holdout -- not
# usable) before this second dataset was chosen instead.
#
# Source: https://www.kaggle.com/datasets/mdtalhask/ai-powered-banking-fraud-detection-dataset-2025
# (Banking_Fraud_Dataset.csv, 10,000 rows, 28.38% fraud rate -- see the
# module docstring in generalizable_scorer.py for why that base rate is
# unrealistically high and what that means for threshold calibration).
#
# Amount is treated as USD (no currency column, but Credit_Score's 300-850
# range is the US FICO scale, and prepare_data.py's Tier 1 conversion
# standardizes this whole domain on USD -- see that file for the rationale).
#
# Usage (from the prism/ repo root):
#   python domains/banking/fraud/data/prepare_tier2_data.py
# Requires: domains/banking/fraud/data/tier2_raw/Banking_Fraud_Dataset.csv
# (not committed -- download via `kaggle datasets download -d
# mdtalhask/ai-powered-banking-fraud-detection-dataset-2025`, unzip into
# that folder).

import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
RAW_PATH = _HERE / "tier2_raw" / "Banking_Fraud_Dataset.csv"
TRAIN_PATH = _HERE / "tier2_train.csv"
HOLDOUT_PATH = _HERE / "tier2_test_holdout.csv"

print("Loading Tier 2 raw data ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")

# ── Derive 'hour' from Transaction_Time, same convention as Tier 1 ──────
df["hour"] = pd.to_datetime(df["Transaction_Time"]).dt.hour

# ── Simple random split (no temporal ordering here, unlike Tier 1 --
# this dataset's Transaction_Time values aren't a meaningful sequence to
# split on the way creditcard.csv's elapsed-seconds Time is) ────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
cut = int(len(df) * 0.70)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

print(f"\n  Train: {len(train_df):,} rows, {train_df.Is_Fraud.mean():.2%} fraud rate")
print(f"  Test:  {len(test_df):,} rows, {test_df.Is_Fraud.mean():.2%} fraud rate")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")
