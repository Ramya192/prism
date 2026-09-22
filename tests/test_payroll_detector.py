"""
tests/test_payroll_detector.py
Tests for PayrollDetectorAgent — the CSV payroll-register rules->tiered
ML->LLM detector, mirroring test_fraud_pipeline.py's structure for
bfsi_fraud. See payroll_detector_agent.py's module docstring for what
Tier 1/2/3 mean here.

A dummy OPENAI_API_KEY (any string) is enough for all of these: the agent
constructs a real ChatOpenAI client eagerly but none of these tests reach
an actual .invoke() call (rules/ML resolve every case here). No network
needed.
"""

import pytest

from domains.payroll_hr.payroll.agents.payroll_detector_agent import PayrollDetectorAgent


@pytest.fixture(scope="module")
def agent():
    return PayrollDetectorAgent(name="TestPayrollDetector")


# Tier 1 schema: SF Salaries' convention (BasePay/OvertimePay/OtherPay/Stated_TotalPay)
TIER1_GOOD = {
    "BasePay": 90000, "OvertimePay": 10000, "OtherPay": 2000, "Benefits": 20000,
    "Stated_TotalPay": 102000, "Stated_TotalPayBenefits": 122000, "JobTitle": "Engineer",
}
TIER1_IMPLAUSIBLE_SPLIT = {
    "BasePay": 5000, "OvertimePay": 95000, "OtherPay": 0, "Benefits": 5000,
    "Stated_TotalPay": 100000, "Stated_TotalPayBenefits": 105000, "JobTitle": "Clerk",
}

# Tier 2 schema: GROSS/Deduction/Net_Pay convention
TIER2_GOOD = {"GROSS": 2000.0, "Deduction": 500.0, "Net_Pay": 1500.0, "Designation": "Software Engineer"}
TIER2_IMPLAUSIBLE_RATIO = {"GROSS": 2000.0, "Deduction": 1900.0, "Net_Pay": 100.0, "Designation": "Software Engineer"}

TIER3_RECORD = {"note": "not payroll shaped"}


class TestSchemaTiering:
    def test_detects_tier1(self):
        assert PayrollDetectorAgent.detect_tier(TIER1_GOOD) == "tier1"

    def test_detects_tier2(self):
        assert PayrollDetectorAgent.detect_tier(TIER2_GOOD) == "tier2"

    def test_detects_tier3_when_no_schema_matches(self):
        assert PayrollDetectorAgent.detect_tier(TIER3_RECORD) == "tier3"

    def test_has_recognizable_schema_matches_tier_result(self):
        assert PayrollDetectorAgent.has_recognizable_schema(TIER1_GOOD)
        assert PayrollDetectorAgent.has_recognizable_schema(TIER2_GOOD)
        assert not PayrollDetectorAgent.has_recognizable_schema(TIER3_RECORD)


class TestRuleBasedFilter:
    """Deterministic reconciliation, one check per tier's own schema --
    exact arithmetic, not a statistical result (see module docstring for
    why no precision/recall/F1 is attached to this layer)."""

    def test_tier1_reconciling_total_is_not_flagged(self):
        assert PayrollDetectorAgent.rule_based_filter(TIER1_GOOD, "tier1") is None

    def test_tier1_mismatched_total_is_flagged(self):
        broken = {**TIER1_GOOD, "Stated_TotalPay": 999999}
        result = PayrollDetectorAgent.rule_based_filter(broken, "tier1")
        assert result is not None and "HIGH" in result and "Reconciliation error" in result

    def test_tier2_reconciling_net_is_not_flagged(self):
        assert PayrollDetectorAgent.rule_based_filter(TIER2_GOOD, "tier2") is None

    def test_tier2_mismatched_net_is_flagged(self):
        broken = {**TIER2_GOOD, "Net_Pay": 999}
        result = PayrollDetectorAgent.rule_based_filter(broken, "tier2")
        assert result is not None and "HIGH" in result and "Reconciliation error" in result

    def test_tier3_returns_none(self):
        assert PayrollDetectorAgent.rule_based_filter(TIER3_RECORD, "tier3") is None


class TestMlFilter:
    """Each tier's scorer catches a pattern rules structurally cannot see
    (verified: 0 rule matches on either tier's labeling scheme's
    holdout) -- an implausible component split for tier1, an implausible
    deduction ratio for tier2."""

    def test_tier1_plausible_split_is_not_flagged_high(self, agent):
        result = agent.ml_filter(TIER1_GOOD, "tier1")
        if result is not None:
            assert "HIGH" not in result

    def test_tier1_implausible_split_is_flagged(self, agent):
        result = agent.ml_filter(TIER1_IMPLAUSIBLE_SPLIT, "tier1")
        assert result is not None and "HIGH" in result

    def test_tier2_plausible_ratio_is_not_flagged_high(self, agent):
        result = agent.ml_filter(TIER2_GOOD, "tier2")
        if result is not None:
            assert "HIGH" not in result

    def test_tier2_implausible_ratio_is_flagged(self, agent):
        result = agent.ml_filter(TIER2_IMPLAUSIBLE_RATIO, "tier2")
        assert result is not None and "HIGH" in result


class TestAnalyse:
    def test_cases_reviewed_increments(self, agent):
        before = agent.cases_reviewed
        agent.analyse(TIER1_GOOD)
        assert agent.cases_reviewed == before + 1

    def test_tier1_reconciliation_break_resolved_without_reaching_llm(self, agent):
        broken = {**TIER1_GOOD, "Stated_TotalPay": 999999}
        result = agent.analyse(broken)
        assert "Reconciliation error" in result

    def test_tier2_reconciliation_break_resolved_without_reaching_llm(self, agent):
        broken = {**TIER2_GOOD, "Net_Pay": 999}
        result = agent.analyse(broken)
        assert "Reconciliation error" in result

    def test_tier3_record_falls_through_to_llm_gracefully(self, agent):
        """No real API key in tests -- must degrade to the safe fallback,
        not crash, and must not fabricate a reconciliation-style verdict
        for data that was never checked."""
        result = agent.analyse(TIER3_RECORD)
        assert "Reconciliation error" not in result
        assert "Risk Level" in result and "Action" in result


class TestResponseParsing:
    def test_parse_response_extracts_fields(self):
        response = "Risk Level: HIGH\nReason: Bad split.\nAction: BLOCK"
        parsed = PayrollDetectorAgent.parse_response(response)
        assert parsed == {"risk_level": "HIGH", "action": "BLOCK", "reason": "Bad split."}

    def test_is_fraud_true_for_medium_and_high(self):
        assert PayrollDetectorAgent.is_fraud({"risk_level": "MEDIUM"})
        assert PayrollDetectorAgent.is_fraud({"risk_level": "HIGH"})

    def test_is_fraud_false_for_low(self):
        assert not PayrollDetectorAgent.is_fraud({"risk_level": "LOW"})
