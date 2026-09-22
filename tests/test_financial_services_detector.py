"""
tests/test_financial_services_detector.py
Tests for FintechDetectorAgent -- the non-bank crypto/digital-asset
rules->tiered ML->LLM detector. Tier 1 is blockchain scam detection,
Tier 2 is exchange/DEX market-manipulation detection (wash trading/
pump-and-dump/ramping) -- see fintech_detector_agent.py's module
docstring for why no rule-layer fraud heuristic exists for either tier
(only a malformed-input check) -- every candidate heuristic measured too
weak on the real data to qualify as a rule.

A dummy OPENAI_API_KEY (any string) is enough for all of these -- no
network needed, rules resolve every case here (the ML layer isn't
exercised by rule-resolved cases, same convention as every other
detector's test file).
"""

import pytest

from domains.financial_services.agents.fintech_detector_agent import FintechDetectorAgent


@pytest.fixture(scope="module")
def agent():
    return FintechDetectorAgent(name="TestFintechDetector")


# Tier 1 schema: crypto scam transaction convention
TIER1_GOOD = {
    "transaction_amount_usd": 929.39, "sender_wallet_age_days": 469, "is_cross_chain": 0,
    "receiver_wallet_age_days": 769, "gas_fee_usd": 2.76, "num_prev_transactions_sender": 1120,
    "num_prev_transactions_receiver": 960, "avg_txn_interval_sender_min": 100.5,
    "failed_txn_ratio_sender": 0.18, "velocity_score": -0.12, "anomaly_score": 0.11,
    "blockchain": "BSC", "transaction_type": "Mint", "token_type": "ERC20", "platform": "Coinbase",
}

# Tier 2 schema: crypto exchange wash-trading convention
TIER2_GOOD = {
    "notional_usd": 5000.0, "round_trip_score": 0.3, "counterparty_reuse_ratio": 0.05,
    "token_liquidity_score": 0.8, "is_dex": 0, "base_amount": 100.0, "price_usd": 50.0,
    "gas_fee_usd": 1.2, "wallet_age_days": 400, "new_wallet_flag": 0,
    "wallet_txn_count_total": 500, "time_since_prev_tx_sec": 3600, "tx_burst_count_5m": 1,
    "counterparty_diversity": 0.9, "pair_txn_count": 10, "token_volume_zscore_1h": 0.5,
    "price_deviation_vwap_pct": 0.01, "notional_zscore_wallet": 0.0, "hour_of_day": 14,
    "is_night": 0, "venue_type": "CEX", "token_is_major": 1,
}

NO_SCHEMA_RECORD = {"note": "not a transaction"}


class TestSchemaTiering:
    def test_detects_tier1(self):
        assert FintechDetectorAgent.detect_tier(TIER1_GOOD) == "tier1"

    def test_detects_tier2(self):
        assert FintechDetectorAgent.detect_tier(TIER2_GOOD) == "tier2"

    def test_detects_tier3_when_no_schema_matches(self):
        assert FintechDetectorAgent.detect_tier(NO_SCHEMA_RECORD) == "tier3"

    def test_has_recognizable_schema_matches_tier_result(self):
        assert FintechDetectorAgent.has_recognizable_schema(TIER1_GOOD)
        assert FintechDetectorAgent.has_recognizable_schema(TIER2_GOOD)
        assert not FintechDetectorAgent.has_recognizable_schema(NO_SCHEMA_RECORD)


class TestRuleBasedFilter:
    def test_missing_amount_blocks_tier1(self):
        record = {k: v for k, v in TIER1_GOOD.items() if k != "transaction_amount_usd"}
        result = FintechDetectorAgent.rule_based_filter(record, "tier1")
        assert result is not None
        assert "no transaction amount" in result

    def test_negative_amount_blocks_tier1(self):
        record = dict(TIER1_GOOD, transaction_amount_usd=-500)
        result = FintechDetectorAgent.rule_based_filter(record, "tier1")
        assert result is not None
        assert "Negative transaction amount" in result

    def test_negative_amount_blocks_tier2(self):
        record = dict(TIER2_GOOD, notional_usd=-50)
        result = FintechDetectorAgent.rule_based_filter(record, "tier2")
        assert result is not None
        assert "Negative transaction amount" in result

    def test_valid_amount_passes_rules(self):
        assert FintechDetectorAgent.rule_based_filter(TIER1_GOOD, "tier1") is None
        assert FintechDetectorAgent.rule_based_filter(TIER2_GOOD, "tier2") is None


class TestParsing:
    def test_parses_well_formed_response(self):
        response = "Risk Level: HIGH\nReason: Multiple weak signals combined.\nAction: BLOCK"
        parsed = FintechDetectorAgent.parse_response(response)
        assert parsed == {"risk_level": "HIGH", "reason": "Multiple weak signals combined.", "action": "BLOCK"}

    def test_is_fraud_treats_unknown_as_fraud(self):
        assert FintechDetectorAgent.is_fraud({"risk_level": "UNKNOWN"})
        assert FintechDetectorAgent.is_fraud({"risk_level": "MEDIUM"})
        assert not FintechDetectorAgent.is_fraud({"risk_level": "LOW"})


class TestAnalyse:
    def test_malformed_tier1_record_resolves_at_rules_layer(self, agent):
        record = dict(TIER1_GOOD, transaction_amount_usd=-100)
        response = agent.analyse(record)
        parsed = FintechDetectorAgent.parse_response(response)
        assert parsed["risk_level"] == "HIGH"
        assert FintechDetectorAgent.is_fraud(parsed)

    def test_malformed_tier2_record_resolves_at_rules_layer(self, agent):
        record = dict(TIER2_GOOD, notional_usd=-50)
        response = agent.analyse(record)
        parsed = FintechDetectorAgent.parse_response(response)
        assert parsed["risk_level"] == "HIGH"
        assert FintechDetectorAgent.is_fraud(parsed)
