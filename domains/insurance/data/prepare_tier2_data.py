# domains/insurance/data/prepare_tier2_data.py
# Builds Tier 2's train/holdout split -- a SECOND, genuinely differently-
# shaped claims schema than Tier 1's (nudratabbas' Claim_Amount/
# Approved_Amount/Days_Between_Service_and_Claim), so the detector can
# recognize more than one real claims-export convention.
#
# Source: https://www.kaggle.com/datasets/tejalaveti2306/health-insurance-claims-data-for-fraud-detection
# (synthetic_health_claims.csv, 20,100 rows, real per-claim Is_Fraudulent
# label, 24.9% fraud rate, no missing values). Verified learnable before
# adopting: AUC 0.857, F1 0.667 on a random holdout, signal spread across
# 5+ features (no single dominant giveaway) -- comparable caliber to
# banking fraud's own Tier 2.
#
# HONEST CAVEAT, kept rather than hidden: this dataset's Claim_Amount
# range ($5,517-$2,499,689, mean ~$1.24M) is unrealistic for individual
# healthcare claims -- roughly 200-1000x larger than Tier 1's
# nudratabbas dataset (which averages ~$570). This is NOT a currency
# issue (both datasets are already USD -- Patient_State/Provider_State
# are US state codes in both, no conversion needed, see
# docs/DATA_CONVENTIONS.md) -- it's simply an implausible numeric range
# in this synthetic generator, similar in spirit to banking fraud Tier 2's
# unrealistically high 28% fraud base rate (mdtalhask). Two candidate
# datasets were searched and rejected before falling back to this one:
# a re-upload of the already-rejected provider-level Medicare dataset,
# and a toy graph-database demo dataset too small to train on. This was
# the best learnable, genuinely-differently-shaped, real-labeled
# candidate found.
#
# Usage (from the prism/ repo root):
#   python domains/insurance/data/prepare_tier2_data.py
# Requires: domains/insurance/data/tier2_raw/synthetic_health_claims.csv
# (not committed -- download via `kaggle datasets download -d
# tejalaveti2306/health-insurance-claims-data-for-fraud-detection`,
# unzip into that folder).

import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
RAW_PATH = _HERE / "tier2_raw" / "synthetic_health_claims.csv"
TRAIN_PATH = _HERE / "tier2_train.csv"
HOLDOUT_PATH = _HERE / "tier2_test_holdout.csv"
SEED = 42

print("Loading Tier 2 raw data ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
cut = int(len(df) * 0.70)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

print(f"\n  Train: {len(train_df):,} rows, {train_df.Is_Fraudulent.mean():.2%} fraud rate")
print(f"  Test:  {len(test_df):,} rows, {test_df.Is_Fraudulent.mean():.2%} fraud rate")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")
