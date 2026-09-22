# domains/payroll_hr/payroll/data/prepare_tier_ml_data.py
# Builds a labeled dataset for payroll's ML anomaly tier from REAL public
# payroll data + SYNTHETIC injected labels -- see anomaly_scorer.py's
# module docstring for why this split exists (real salaries are public
# record; a labeled payroll-fraud dataset structurally doesn't exist
# because real payroll data is confidential PII no organization publishes).
#
# Source: https://www.kaggle.com/datasets/kaggle/sf-salaries (public San
# Francisco city employee compensation, 2011-2014, CC0).
#
# IMPORTANT — this is the SECOND labeling scheme tried, not the first.
# The first (reconciliation-break injection: silently drop overtime,
# inflate the stated total, etc.) was measured and rejected: a
# deterministic rules check gets PERFECT precision/recall/F1 on that
# labeling, because it's exactly what reconciliation math is designed to
# catch — the ML tier only reached AUC 0.77 there, strictly worse at its
# own dataset's task. Redundant with rules, not complementary.
#
# This scheme instead REDISTRIBUTES an unchanged, correctly-reconciling
# total across BasePay/OvertimePay/OtherPay into an implausible split
# (e.g. overtime far exceeding base pay) — Stated_TotalPay is untouched,
# so rule_based_filter()'s reconciliation check CANNOT see anything wrong
# (by construction, it always passes). Only a model that has learned what
# a realistic component split looks like can catch it. This is the
# genuinely complementary job described in anomaly_scorer.py.
#
# Compensation TOTALS are always real; for flagged rows, the SPLIT among
# components is synthetic and known; for unflagged rows, everything is
# the real, untouched original data.
#
# Usage (from the prism/ repo root):
#   python domains/payroll_hr/payroll/data/prepare_tier_ml_data.py
# Requires: domains/payroll_hr/payroll/data/tier_ml_raw/Salaries.csv (not committed --
# download via `kaggle datasets download -d kaggle/sf-salaries`, unzip
# into that folder).

import numpy as np
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
RAW_PATH = _HERE / "tier_ml_raw" / "Salaries.csv"
TRAIN_PATH = _HERE / "tier_ml_train.csv"
HOLDOUT_PATH = _HERE / "tier_ml_test_holdout.csv"
SEED = 42

print("Loading SF Salaries raw data ...")
df = pd.read_csv(RAW_PATH, low_memory=False)
for c in ["BasePay", "OvertimePay", "OtherPay", "Benefits", "TotalPay", "TotalPayBenefits"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Benefits wasn't reported for early years -- treat missing as 0, drop rows
# missing anything else (need BasePay/OvertimePay/OtherPay/TotalPay(Benefits)
# to compute a ground-truth reconciliation at all).
df["Benefits"] = df["Benefits"].fillna(0)
df = df.dropna(subset=["BasePay", "OvertimePay", "OtherPay", "TotalPay", "TotalPayBenefits"]).copy()
df = df[df["BasePay"] > 0].reset_index(drop=True)  # drop $0-base rows, not representative
print(f"  Usable rows: {len(df):,}")

rng = np.random.default_rng(SEED)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle before labeling

# ── Redistribute ~50% of rows' real, unchanged total into an implausible
# component split (label 1) -- the other half keep their real, untouched
# split (label 0). Stated_TotalPay/Stated_TotalPayBenefits are IDENTICAL
# to the real TotalPay/TotalPayBenefits for every row, flagged or not --
# reconciliation math has nothing to catch here by construction. Only the
# BasePay/OvertimePay/OtherPay split differs for flagged rows.
n = len(df)
flag_idx = rng.choice(n, size=n // 2, replace=False)
is_flagged = np.zeros(n, dtype=bool)
is_flagged[flag_idx] = True

df["Stated_TotalPay"] = df["TotalPay"]
df["Stated_TotalPayBenefits"] = df["TotalPayBenefits"]

new_base = df["BasePay"].to_numpy(dtype=float).copy()
new_ot = df["OvertimePay"].to_numpy(dtype=float).copy()
new_other = df["OtherPay"].to_numpy(dtype=float).copy()

redistribution_kind = rng.choice(["overtime_heavy", "other_heavy", "base_only"], size=n)

for i in np.where(is_flagged)[0]:
    total = df.loc[i, "TotalPay"]
    kind = redistribution_kind[i]
    if total <= 0:
        continue
    if kind == "overtime_heavy":
        # Overtime pay alone exceeds base pay -- implausible for almost any
        # real role (overtime is a multiplier ON a base, not the majority
        # of pay), but arithmetically the total is untouched.
        new_ot[i] = total * rng.uniform(0.6, 0.9)
        new_base[i] = total - new_ot[i]
        new_other[i] = 0.0
    elif kind == "other_heavy":
        # "Other pay" (bonuses/misc) dominating over a base salary is a
        # similarly unrealistic shape for routine compensation.
        new_other[i] = total * rng.uniform(0.6, 0.9)
        new_base[i] = total - new_other[i]
        new_ot[i] = 0.0
    else:  # base_only
        # A real row with genuine overtime/other pay reassigned entirely
        # into base salary -- erases a real, normally-present pattern.
        new_base[i] = total
        new_ot[i] = 0.0
        new_other[i] = 0.0

df["BasePay"] = new_base
df["OvertimePay"] = new_ot
df["OtherPay"] = new_other
df["Is_Flagged"] = is_flagged.astype(int)

cut = int(n * 0.70)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

print(f"  Train: {len(train_df):,} rows, {train_df.Is_Flagged.mean():.1%} flagged")
print(f"  Test:  {len(test_df):,} rows, {test_df.Is_Flagged.mean():.1%} flagged")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")
