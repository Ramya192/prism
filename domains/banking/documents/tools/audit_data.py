# domains/banking/documents/tools/audit_data.py
# Ported from document_intelligence_system/tools/audit_data.py — logic
# unchanged, path updated. Data-quality checks on the raw Kaggle CSV that
# generate_statement.py builds statements from — this is what originally
# surfaced the ATM+Credit mismatch that clean_transactions() in
# generate_statement.py now fixes.
#
# Usage (from the prism/ repo root):
#   python domains/banking/documents/tools/audit_data.py

import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "raw" / "bank_transactions_data_2.csv"

df = pd.read_csv(CSV_PATH)

print("=== SHAPE ===")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

print("\n=== NULL VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATE TRANSACTION IDs ===")
dup_count = df["TransactionID"].duplicated().sum()
print(f"Duplicate TxnIDs: {dup_count}")

print("\n=== TRANSACTION TYPE VALUES ===")
print(df["TransactionType"].value_counts())

print("\n=== CHANNEL VALUES ===")
print(df["Channel"].value_counts())

print("\n=== ATM + CREDIT MISMATCH ===")
atm_credit = df[(df["Channel"] == "ATM") & (df["TransactionType"] == "Credit")]
print(f"ATM transactions marked as Credit: {len(atm_credit)}")

print("\n=== ZERO OR NEGATIVE AMOUNTS ===")
print(f'Zero amounts: {(df["TransactionAmount"] == 0).sum()}')
print(f'Negative amounts: {(df["TransactionAmount"] < 0).sum()}')

print("\n=== NEGATIVE ACCOUNT BALANCE ===")
print(f'Negative balances: {(df["AccountBalance"] < 0).sum()}')

print("\n=== AMOUNT STATS ===")
print(df["TransactionAmount"].describe())

print("\n=== UNKNOWN MERCHANT IDs ===")
known = [f"M{str(i).zfill(3)}" for i in range(1, 100)]
unknown = df[~df["MerchantID"].isin(known)]["MerchantID"].value_counts()
print(f"Unknown merchant IDs: {len(unknown)}")
print(unknown.head(10))

print("\n=== DATE ISSUES ===")
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
print(f'Min date: {df["TransactionDate"].min()}')
print(f'Max date: {df["TransactionDate"].max()}')
print(f'Future dates: {(df["TransactionDate"] > pd.Timestamp.now()).sum()}')

print("\n=== ONLINE + NO MERCHANT ===")
online_no_merchant = df[(df["Channel"] == "Online") & (df["MerchantID"].isna())]
print(f"Online transactions with no MerchantID: {len(online_no_merchant)}")

print("\n=== LOGIN ATTEMPTS > 3 (suspicious) ===")
print(f'High login attempts (>3): {(df["LoginAttempts"] > 3).sum()}')

print("\n=== TRANSACTION DURATION OUTLIERS ===")
print(df["TransactionDuration"].describe())
print(f'Duration > 300s: {(df["TransactionDuration"] > 300).sum()}')
