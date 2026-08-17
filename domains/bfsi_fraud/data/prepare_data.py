# domains/bfsi_fraud/data/prepare_data.py
# Ported from fraud_detection/data/prepare_data.py — logic unchanged, paths
# updated to this domain's own data/ folder.
#
# Rebuilds all derived datasets from the original creditcard.csv:
#   - train.csv                (70% temporal window, full imbalance, all features)
#   - test_holdout.csv         (30% temporal window, NEVER trained on, all features)
#   - transactions_balanced.csv (small 50/50 demo set, sampled ONLY from the train
#     window — small enough to ship in git so main.py/streamlit_app.py work without
#     downloading the full Kaggle CSV, and this is also what agents/detector_agent.py's
#     MLScorer trains on by default, so it must never contain holdout rows)
#
# Keeps ALL columns: Time, V1-V28, Amount, is_Fraud, hour
# Run this once (from the prism/ repo root) before running main.py
#
# Usage: python domains/bfsi_fraud/data/prepare_data.py
# Requires: download creditcard.csv from
#   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#   and place it at domains/bfsi_fraud/data/creditcard.csv — not committed
#   (150MB, too large for GitHub, and not needed for the demo out of the box).

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
_HERE             = Path(__file__).parent
RAW_PATH           = _HERE / "creditcard.csv"
TRAIN_PATH         = _HERE / "train.csv"
HOLDOUT_PATH       = _HERE / "test_holdout.csv"
BALANCED_PATH      = _HERE / "transactions_balanced.csv"
BALANCED_SEED      = 42

# ── Load original dataset ──────────────────────────────────────────────
print("Loading original creditcard.csv ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")          # (284807, 31)
print(f"  Columns: {list(df.columns)}")

# ── Rename Class → is_Fraud if needed ─────────────────────────────────
if "Class" in df.columns and "is_Fraud" not in df.columns:
    df = df.rename(columns={"Class": "is_Fraud"})
    print("  Renamed 'Class' → 'is_Fraud'")

# ── Add hour feature from Time column ─────────────────────────────────
# Time = seconds elapsed since first transaction in dataset
# hour = which hour of the day (0-23) using modulo
df["hour"] = (df["Time"] // 3600 % 24).astype(int)
print("  Added 'hour' feature")

# ── Temporal split BEFORE any balancing ────────────────────────────────
df = df.sort_values("Time").reset_index(drop=True)
cut = int(len(df) * 0.70)

train_df = df.iloc[:cut].copy()
test_df  = df.iloc[cut:].copy()      # holdout: NEVER balanced, NEVER trained on

print(f"\n  Train window: t={train_df.Time.min():.0f}–{train_df.Time.max():.0f} "
      f"({len(train_df):,} txns, {train_df.is_Fraud.sum()} fraud, "
      f"{train_df.is_Fraud.mean():.4%} base rate)")
print(f"  Test  window: t={test_df.Time.min():.0f}–{test_df.Time.max():.0f} "
      f"({len(test_df):,} txns, {test_df.is_Fraud.sum()} fraud, "
      f"{test_df.is_Fraud.mean():.4%} base rate)")

# NOTE: we deliberately do NOT downsample the training set.
# Downsampling to 492 legit rows discards ~99.8% of legitimate behaviour.
# class_weight="balanced" in the RandomForest handles imbalance instead.
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")

# ── Small 50/50 demo dataset — sampled ONLY from train_df ──────────────
# This file ships in git (small enough for GitHub) so the CLI demo
# (main.py) and Streamlit UI (streamlit_app.py) work out of the box without
# downloading the full Kaggle CSV. Critically, it must be sampled from
# train_df, not the full df: agents/detector_agent.py's MLScorer trains
# directly on this file, so if it included any test_df rows the deployed
# system's ML fast-path would be evaluated (via eval_harness.py's
# --with-llm-sample run) on data it had partially seen — the exact
# leakage bug this script exists to prevent, just reintroduced one layer
# up. Sampling from train_df alone keeps the holdout genuinely unseen.
fraud_pool = train_df[train_df["is_Fraud"] == 1]
legit_pool = train_df[train_df["is_Fraud"] == 0]
n_balanced = min(len(fraud_pool), 492)  # cap at how much fraud the original demo used

balanced_df = pd.concat(
    [
        fraud_pool.sample(n=n_balanced, random_state=BALANCED_SEED),
        legit_pool.sample(n=n_balanced, random_state=BALANCED_SEED),
    ]
).sample(frac=1, random_state=BALANCED_SEED).reset_index(drop=True)  # shuffle

balanced_df.to_csv(BALANCED_PATH, index=False)
print(f"✓ Wrote {BALANCED_PATH} ({len(balanced_df)} txns, "
      f"{n_balanced} fraud + {n_balanced} legit, sampled from the train window only)")
