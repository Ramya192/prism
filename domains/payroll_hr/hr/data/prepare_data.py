# domains/payroll_hr/hr/data/prepare_data.py
# Builds the train/holdout split for hr's fraud detector -- fraudulent
# JOB POSTINGS (a real, well-studied HR-adjacent fraud problem:
# recruitment scams), not employee-level attendance/timesheet fraud.
#
# Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
# (the EMSCAD dataset, University of the Aegean -- REAL, not synthetic,
# unlike most datasets used in this repo so far). 17,880 postings, real
# per-POSTING fraudulent label, 4.84% fraud rate.
#
# This is the SECOND dataset considered for this domain. The originally
# planned candidate (alnahiyan/employee-attendance-fraud-and-anomaly-detection)
# was rejected: its user_labels.csv is explicitly documented (in the
# dataset's own README) as USER-level risk labels, the same
# entity-vs-artifact conflict that got rohitrox rejected for healthcare.
# This dataset's label is per-posting instead -- scoring the artifact
# (is THIS job posting fraudulent), not the entity that created it.
#
# CURRENCY: deliberately does NOT use salary_range as a feature. This is
# a global, multi-country dataset (locations span US/NZ/Germany/etc. with
# no currency field), and salary_range is 84% missing besides, with mixed
# units even where present ("20000-28000" vs "60-80", likely annual vs
# hourly). Rather than force an unreliable currency normalization onto a
# field that's mostly absent anyway, this uses PRESENCE of a salary range
# as a currency-agnostic structural signal instead -- a real, published,
# strong predictor for this exact dataset (scammers often omit concrete
# salary info), verified learnable this way: AUC 0.968, F1 0.647, no
# single dominant feature (has_company_profile tops out at 19%
# importance). See docs/DATA_CONVENTIONS.md.
#
# Usage (from the prism/ repo root):
#   python domains/payroll_hr/hr/data/prepare_data.py
# Requires: domains/payroll_hr/hr/data/raw/fake_job_postings.csv (not committed --
# download via `kaggle datasets download -d
# shivamb/real-or-fake-fake-jobposting-prediction`, unzip into that folder).

import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
RAW_PATH = _HERE / "raw" / "fake_job_postings.csv"
TRAIN_PATH = _HERE / "train.csv"
HOLDOUT_PATH = _HERE / "test_holdout.csv"
SEED = 42

print("Loading job postings raw data ...")
df = pd.read_csv(RAW_PATH)
print(f"  Original shape: {df.shape}")

# ── Structural/presence features -- deliberately no raw salary amount ──
df["has_salary_range"] = df["salary_range"].notna().astype(int)
df["has_company_profile"] = df["company_profile"].notna().astype(int)
df["has_department"] = df["department"].notna().astype(int)
df["has_requirements"] = df["requirements"].notna().astype(int)
df["has_benefits"] = df["benefits"].notna().astype(int)
df["description_length"] = df["description"].fillna("").str.len()
df["country_code"] = df["location"].fillna("").str.split(",").str[0].str.strip()

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
cut = int(len(df) * 0.70)
train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

print(f"\n  Train: {len(train_df):,} rows, {train_df.fraudulent.mean():.2%} fraud rate")
print(f"  Test:  {len(test_df):,} rows, {test_df.fraudulent.mean():.2%} fraud rate")

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(HOLDOUT_PATH, index=False)
print(f"\n✓ Wrote {TRAIN_PATH} and {HOLDOUT_PATH}")
