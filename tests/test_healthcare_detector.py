"""
tests/test_healthcare_detector.py
Tests for HealthcareDetectorAgent — the claim-level rules->tiered
ML->LLM detector, mirroring test_fraud_pipeline.py / test_payroll_detector.py's
structure. See healthcare_detector_agent.py's module docstring for what
Tier 1/2/3 mean here.

A dummy OPENAI_API_KEY (any string) is enough for all of these -- no
network needed, rules/ML resolve every case here.
"""

import pytest

from domains.insurance.agents.healthcare_detector_agent import HealthcareDetectorAgent


@pytest.fixture(scope="module")
def agent():
    return HealthcareDetectorAgent(name="TestHealthcareDetector")


# Tier 1 schema: nudratabbas' convention
TIER1_GOOD = {
    "Claim_Amount": 500.0, "Approved_Amount": 400.0, "Days_Between_Service_and_Claim": 15,
    "Number_of_Claims_Per_Provider_Monthly": 30, "Length_of_Stay": 2, "Chronic_Condition_Flag": 1,
    "Prior_Visits_12m": 3, "Patient_Age": 55, "Patient_Gender": "Male", "Insurance_Type": "Medicaid",
    "Provider_Specialty": "Cardiology", "Visit_Type": "Outpatient",
}

# Tier 2 schema: tejalaveti2306's convention
TIER2_GOOD = {
    "Claim_Amount": 1200000.0, "Policy_Number": "XAI123", "Provider_Patient_Distance_Miles": 5,
    "Number_of_Procedures": 2, "Claim_Date": "2021-06-16", "Service_Date": "2021-05-01",
    "Policy_Expiration_Date": "2024-11-20", "Provider_Type": "Clinic",
}

NO_SCHEMA_CLAIM = {"note": "not a claim"}


class TestSchemaTiering:
    def test_detects_tier1(self):
        assert HealthcareDetectorAgent.detect_tier(TIER1_GOOD) == "tier1"

    def test_detects_tier2(self):
        assert HealthcareDetectorAgent.detect_tier(TIER2_GOOD) == "tier2"

    def test_detects_tier3_when_no_schema_matches(self):
        assert HealthcareDetectorAgent.detect_tier(NO_SCHEMA_CLAIM) == "tier3"

    def test_has_recognizable_schema_matches_tier_result(self):
        assert HealthcareDetectorAgent.has_recognizable_schema(TIER1_GOOD)
        assert HealthcareDetectorAgent.has_recognizable_schema(TIER2_GOOD)
        assert not HealthcareDetectorAgent.has_recognizable_schema(NO_SCHEMA_CLAIM)


class TestRuleBasedFilter:
    """Genuine data-integrity/policy impossibilities -- not a statistical
    model, so no precision/recall/F1 claim is attached to either tier's
    rules layer. Both tiers' clean training data trigger none of these
    (verified), same honest situation as payroll Tier 2 -- kept for
    correctness against malformed real input regardless."""

    def test_tier1_normal_claim_is_not_flagged(self):
        assert HealthcareDetectorAgent.rule_based_filter(TIER1_GOOD, "tier1") is None

    def test_tier1_approved_exceeding_claimed_is_flagged(self):
        bad = {**TIER1_GOOD, "Approved_Amount": 9999999.0}
        result = HealthcareDetectorAgent.rule_based_filter(bad, "tier1")
        assert result is not None and "HIGH" in result and "exceeds the claimed amount" in result

    def test_tier1_negative_days_is_flagged(self):
        bad = {**TIER1_GOOD, "Days_Between_Service_and_Claim": -5}
        result = HealthcareDetectorAgent.rule_based_filter(bad, "tier1")
        assert result is not None and "HIGH" in result

    def test_tier2_normal_claim_is_not_flagged(self):
        assert HealthcareDetectorAgent.rule_based_filter(TIER2_GOOD, "tier2") is None

    def test_tier2_claim_after_policy_expiration_is_flagged(self):
        bad = {**TIER2_GOOD, "Claim_Date": "2025-01-01"}
        result = HealthcareDetectorAgent.rule_based_filter(bad, "tier2")
        assert result is not None and "HIGH" in result and "expiration" in result

    def test_tier2_service_after_claim_is_flagged(self):
        bad = {**TIER2_GOOD, "Service_Date": "2021-07-01"}  # after Claim_Date 2021-06-16
        result = HealthcareDetectorAgent.rule_based_filter(bad, "tier2")
        assert result is not None and "HIGH" in result

    def test_tier3_returns_none(self):
        assert HealthcareDetectorAgent.rule_based_filter(NO_SCHEMA_CLAIM, "tier3") is None


class TestMlFilter:
    def test_tier1_typical_claim_is_not_flagged_high(self, agent):
        result = agent.ml_filter(TIER1_GOOD, "tier1")
        if result is not None:
            assert "HIGH" not in result

    def test_tier1_fast_submission_scores_higher_than_typical(self, agent):
        """Days_Between_Service_and_Claim is Tier 1's dominant feature
        (fraud averages ~3 days vs. ~15 for legitimate claims in
        training)."""
        fast_claim = {**TIER1_GOOD, "Days_Between_Service_and_Claim": 1}
        assert agent.scorer.score(fast_claim) > agent.scorer.score(TIER1_GOOD)

    def test_tier2_typical_claim_is_not_flagged_high(self, agent):
        result = agent.ml_filter(TIER2_GOOD, "tier2")
        if result is not None:
            assert "HIGH" not in result


class TestAnalyse:
    def test_cases_reviewed_increments(self, agent):
        before = agent.cases_reviewed
        agent.analyse(TIER1_GOOD)
        assert agent.cases_reviewed == before + 1

    def test_tier1_rule_violation_resolved_without_reaching_llm(self, agent):
        bad = {**TIER1_GOOD, "Approved_Amount": 9999999.0}
        result = agent.analyse(bad)
        assert "exceeds the claimed amount" in result

    def test_tier2_rule_violation_resolved_without_reaching_llm(self, agent):
        bad = {**TIER2_GOOD, "Claim_Date": "2025-01-01"}
        result = agent.analyse(bad)
        assert "expiration" in result

    def test_no_schema_claim_falls_through_to_llm_gracefully(self, agent):
        result = agent.analyse(NO_SCHEMA_CLAIM)
        assert "exceeds the claimed amount" not in result
        assert "Risk Level" in result and "Action" in result


class TestResponseParsing:
    def test_parse_response_extracts_fields(self):
        response = "Risk Level: HIGH\nReason: Suspicious timing.\nAction: BLOCK"
        parsed = HealthcareDetectorAgent.parse_response(response)
        assert parsed == {"risk_level": "HIGH", "action": "BLOCK", "reason": "Suspicious timing."}

    def test_is_fraud_true_for_medium_and_high(self):
        assert HealthcareDetectorAgent.is_fraud({"risk_level": "MEDIUM"})
        assert HealthcareDetectorAgent.is_fraud({"risk_level": "HIGH"})

    def test_is_fraud_false_for_low(self):
        assert not HealthcareDetectorAgent.is_fraud({"risk_level": "LOW"})
