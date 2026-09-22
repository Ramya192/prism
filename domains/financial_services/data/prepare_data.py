# domains/financial_services/data/prepare_data.py
# Prepares Financial Services' Tier 1 training data: Crypto Scam
# Transaction Dataset (Kaggle: muhammadhussnain09/crypto-scam-transaction-
# dataset) -- per-transaction blockchain records, real is_scam label.
#
# Promoted to Tier 1 after outscoring the original Tier 1 candidate
# (Central Asia Fintech mobile-wallet data, now benched in
# backup_wallet_raw/ -- see that dataset's own honest number, AUC 0.738/
# F1 0.144, in this file's git history) on the same measured, no-leakage
# basis. USD-native throughout (transaction_amount_usd, gas_fee_usd) --
# matches this project's standing USD convention, unlike the benched
# wallet dataset (UZS, never converted).
#
# Two other real candidates were checked and rejected before this one:
#   - M-Pesa (calebboen/mpesa-transactions-fraud, "synthetic" in its own
#     filename): legit rows have ~0 balance-reconciliation error
#     (sender_balance_after == sender_balance_before - amount, to
#     floating-point precision); fraud rows average an $885 discrepancy.
#     A label-generation artifact trivially reconstructible from raw
#     fields, not real fraud signal -- worse than PaySim's own artifact.
#   - UPI Transactions (skullagos5246/upi-transactions-2024-dataset):
#     AUC 0.477, chance-level -- every feature's fraud-vs-legit
#     distribution measured nearly identical. Also a scope concern even
#     if it had passed: explicit sender_bank/receiver_bank fields make
#     it bank-to-bank money movement, arguably Banking's territory, not
#     Financial Services'.
#
# Chosen dataset's own vetting: top features by importance (is_cross_chain,
# sender_wallet_age_days, failed_txn_ratio_sender) checked individually --
# no single feature separates classes near-deterministically (is_cross_chain
# is highest at 34% importance; scam rate 23.7% when cross-chain vs. 3.6%
# when not -- a real, plausible signal, not a giveaway). velocity_score
# and anomaly_score look pre-computed by name but measured weak
# correlation (0.077, 0.008) -- checked, not assumed safe.
#
# Real measured AUC 0.908 / F1 0.425 (RandomForest, test_size=0.2,
# random_state=42) -- see tools/scam_scorer.py for the full number set.

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "domains/financial_services/data/tier1_raw/crypto_scam_transaction_dataset.csv"
OUT_TRAIN = "domains/financial_services/data/train.csv"
OUT_HOLDOUT = "domains/financial_services/data/test_holdout.csv"


def prepare():
    df = pd.read_csv(RAW_PATH)
    train, holdout = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_scam"]
    )
    train.to_csv(OUT_TRAIN, index=False)
    holdout.to_csv(OUT_HOLDOUT, index=False)
    print(f"Prepared {len(train):,} train / {len(holdout):,} holdout rows "
          f"({df['is_scam'].mean():.2%} base rate).")


if __name__ == "__main__":
    prepare()
