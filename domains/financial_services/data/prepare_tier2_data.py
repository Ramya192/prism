# domains/financial_services/data/prepare_tier2_data.py
# Prepares Financial Services' Tier 2 training data: Crypto Exchange
# Fraud & Wash Trading Detection (Kaggle: sergionefedov/crypto-exchange-
# fraud-and-wash-trading-detection) -- crypto_fraud_master.csv,
# per-transaction exchange/DEX records, real is_manipulative label
# (manipulation_type: wash_trade / pump_and_dump / ramping / none).
#
# A genuinely different fraud problem from Tier 1, not a reduced copy of
# it -- Tier 1 asks "is this transfer a scam draining a wallet", this
# asks "is this trade manipulating a market" -- and a genuinely
# different schema (exchange/DEX venue, liquidity, price-deviation-from-
# VWAP features vs. Tier 1's wallet-age/cross-chain-bridge features).
# USD-native (notional_usd, gas_fee_usd), same as Tier 1.
#
# Checked for leakage before trusting it, same as every dataset in this
# project: many pre-engineered "_score"/"_zscore" fields here look like
# they could already encode the label (round_trip_score,
# counterparty_reuse_ratio, token_volume_zscore_1h, notional_zscore_wallet)
# -- measured correlation with is_manipulative individually, max was
# 0.134 (counterparty_diversity), nowhere near a giveaway. No candidate
# rule reached a rule's bar either (best 2-condition combo tried: ~0%
# hit rate on only 7 rows -- meaningless; round_trip_score>0.99 alone:
# 7.2% manipulative rate, barely above the 5.9% base rate) -- Tier 2's
# rules layer is malformed-input-only, same honest gap as Tier 1's.
#
# Real measured AUC 0.928 / F1 0.356 (RandomForest, test_size=0.2,
# random_state=42) -- see tools/wash_trading_scorer.py for the full
# number set.

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "domains/financial_services/data/tier2_raw/crypto_fraud_master.csv"
OUT_TRAIN = "domains/financial_services/data/tier2_train.csv"
OUT_HOLDOUT = "domains/financial_services/data/tier2_test_holdout.csv"


def prepare():
    df = pd.read_csv(RAW_PATH)
    train, holdout = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_manipulative"]
    )
    train.to_csv(OUT_TRAIN, index=False)
    holdout.to_csv(OUT_HOLDOUT, index=False)
    print(f"Prepared {len(train):,} train / {len(holdout):,} holdout rows "
          f"({df['is_manipulative'].mean():.2%} base rate).")


if __name__ == "__main__":
    prepare()
