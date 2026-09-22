# domains/payroll_hr/payroll/data/prepare_tier2_data.py
# Builds Tier 2's labeled dataset -- a SECOND, genuinely differently-shaped
# payroll schema than Tier 1's (SF Salaries' BasePay/OvertimePay/OtherPay/
# TotalPay), so the CSV-register detector can recognize more than one
# real payroll-export convention. See anomaly_scorer.py's module
# docstring for how this fits into the tiering.
#
# Source: https://www.kaggle.com/datasets/jb1433/sample-employees-monthly-salary
# (Employee_monthly_salary.csv, 1,802 rows, CC0) -- GROSS/Deduction/Net_Pay,
# a simpler two-component schema (no overtime/other breakdown) than Tier 1's.
# Verified: 100% of rows already satisfy GROSS - Deduction == Net_Pay exactly.
#
# CURRENCY: this dataset has no currency field, but its job titles (Software
# Engineer, Solutions Architect -- Indian IT-services "Designation" style)
# and monthly figures (median ~150,330 native units/month) only make sense
# as INR -- as USD that would be an implausible $1.8M/year for those roles;
# converted, it's a realistic ~$24K/year. Converted here to USD (2020
# average rate, matching this dataset's publish date: 1 USD = 74.160 INR,
# per x-rates.com's 2020 average) for consistency with Tier 1's SF Salaries
# (USD) and bfsi_fraud's own EUR->USD conversion -- same reasoning as
# prepare_data.py's conversion there: keep every dataset in this domain,
# and its rules layer, agreeing on one currency.
#
# Labeling: unlike Tier 1's redistribution-across-components approach (this
# schema only has ONE deduction figure, nothing to redistribute), flagged
# rows here get an implausible Deduction-to-Gross ratio injected (near-0%
# for a high earner, or >90%) with Net_Pay recomputed to match exactly --
# GROSS - Deduction == Net_Pay holds for every row, flagged or not, so
# rules cannot see this pattern; only a model that has learned what a
# realistic deduction rate looks like can.
#
# Usage (from the prism/ repo root):
#   python domains/payroll_hr/payroll/data/prepare_tier2_data.py
# Requires: domains/payroll_hr/payroll/data/tier2_raw/Employee_monthly_salary.csv
# (not committed -- download via `kaggle datasets download -d
# jb1433/sample-employees-monthly-salary`, unzip into that folder).

import numpy as np
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
RAW_PATH = _HERE / "tier2_raw" / "Employee_monthly_salary.csv"
TRAIN_PATH = _HERE / "tier2_train.csv"
HOLDOUT_PATH = _HERE / "tier2_test_holdout.csv"
SEED = 42
# See docs/DATA_CONVENTIONS.md for the general rule this follows.
INR_TO_USD_2020 = 1 / 74.160

print("Loading Tier 2 raw data ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")

df["GROSS"] = (df["GROSS"] * INR_TO_USD_2020).round(2)
df["Deduction"] = (df["Deduction"] * INR_TO_USD_2020).round(2)
df["Net_Pay"] = df["GROSS"] - df["Deduction"]  # derived, not independently rounded -- keeps the identity exact
print(f"  Converted GROSS/Deduction INR -> USD at {INR_TO_USD_2020:.6f} (1 USD = 74.160 INR, 2020 average); Net_Pay re-derived")

diff = (df["GROSS"] - df["Deduction"] - df["Net_Pay"]).abs()
assert (diff < 0.01).all(), "Reconciliation should still hold after a uniform currency conversion"

rng = np.random.default_rng(SEED)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

n = len(df)
flag_idx = rng.choice(n, size=n // 2, replace=False)
is_flagged = np.zeros(n, dtype=bool)
is_flagged[flag_idx] = True

new_deduction = df["Deduction"].to_numpy(dtype=float).copy()
new_net = df["Net_Pay"].to_numpy(dtype=float).copy()

deduction_kind = rng.choice(["near_zero", "extreme_high"], size=n)

for i in np.where(is_flagged)[0]:
    gross = df.loc[i, "GROSS"]
    if deduction_kind[i] == "near_zero":
        # Real data's 25th percentile is ~14.9% and its min is 0% -- an
        # injected band of 0-8% deliberately overlaps the real
        # distribution's lower tail instead of sitting safely outside it,
        # so this isn't a trivially-separable task.
        new_deduction[i] = gross * rng.uniform(0.0, 0.08)
    else:
        # Real data's 75th percentile is ~34.7%, max 86.4% -- 55-75%
        # overlaps the real upper tail rather than sitting past its max.
        new_deduction[i] = gross * rng.uniform(0.55, 0.75)
    new_net[i] = gross - new_deduction[i]

df["Deduction"] = new_deduction
df["Net_Pay"] = new_net
df["Deduction_percentage"] = (df["Deduction"] / df["GROSS"] * 100).round(2)
df["Is_Flagged"] = is_flagged.astype(int)

cut = int(n * 0.70)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

print(f"  Train: {len(train_df):,} rows, {train_df.Is_Flagged.mean():.1%} flagged")
print(f"  Test:  {len(test_df):,} rows, {test_df.Is_Flagged.mean():.1%} flagged")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")
