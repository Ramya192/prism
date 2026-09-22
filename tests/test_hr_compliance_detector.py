"""
tests/test_hr_detector.py
Tests for HrDetectorAgent — the job-posting rules->ML->LLM detector,
mirroring test_healthcare_detector.py's structure. See
hr_detector_agent.py's module docstring for why this domain has only
one tier and why it scores per-posting, not per-company.

A dummy OPENAI_API_KEY (any string) is enough for all of these -- no
network needed, rules/ML resolve every case here.
"""

import pytest

from domains.payroll_hr.hr.agents.hr_detector_agent import HrDetectorAgent


@pytest.fixture(scope="module")
def agent():
    return HrDetectorAgent(name="TestHrDetector")


GOOD_POSTING = {
    "title": "Software Engineer", "description": "Build great software for our platform.",
    "telecommuting": 0, "has_company_logo": 1, "has_questions": 1,
    "company_profile": "We are a great, established company.",
    "salary_range": "80000-100000", "department": "Engineering",
    "requirements": "3+ years experience", "benefits": "Health insurance",
    "employment_type": "Full-time", "industry": "Technology",
    "location": "US, CA, San Francisco",
}
NO_SCHEMA_POSTING = {"note": "not a job posting"}


class TestSchemaCheck:
    def test_recognizes_full_schema(self):
        assert HrDetectorAgent.has_recognizable_schema(GOOD_POSTING)

    def test_rejects_missing_schema(self):
        assert not HrDetectorAgent.has_recognizable_schema(NO_SCHEMA_POSTING)


class TestRuleBasedFilter:
    """Genuine malformed-input checks -- not a restatement of the ML
    signal (which looks at OTHER fields' presence, not title/description
    themselves)."""

    def test_normal_posting_is_not_flagged(self):
        assert HrDetectorAgent.rule_based_filter(GOOD_POSTING) is None

    def test_empty_title_is_flagged(self):
        bad = {**GOOD_POSTING, "title": ""}
        result = HrDetectorAgent.rule_based_filter(bad)
        assert result is not None and "HIGH" in result and "no title" in result

    def test_missing_description_is_flagged(self):
        bad = {**GOOD_POSTING, "description": ""}
        result = HrDetectorAgent.rule_based_filter(bad)
        assert result is not None and "HIGH" in result and "no description" in result

    def test_whitespace_only_title_is_flagged(self):
        bad = {**GOOD_POSTING, "title": "   "}
        result = HrDetectorAgent.rule_based_filter(bad)
        assert result is not None and "HIGH" in result


class TestMlFilter:
    def test_well_documented_posting_is_not_flagged_high(self, agent):
        result = agent.ml_filter(GOOD_POSTING)
        if result is not None:
            assert "HIGH" not in result

    def test_missing_company_info_scores_higher_than_well_documented(self, agent):
        """has_company_profile/has_company_logo are this scorer's top two
        features -- a posting missing both should score meaningfully
        higher than one with both present, even if it doesn't cross the
        HIGH threshold alone."""
        sparse_posting = {
            "title": "Work From Home Opportunity", "description": "Earn money from home.",
            "telecommuting": 1, "has_company_logo": 0, "has_questions": 0,
        }
        assert agent.scorer.score(sparse_posting) > agent.scorer.score(GOOD_POSTING)


class TestAnalyse:
    def test_cases_reviewed_increments(self, agent):
        before = agent.cases_reviewed
        agent.analyse(GOOD_POSTING)
        assert agent.cases_reviewed == before + 1

    def test_rule_violation_resolved_without_reaching_llm(self, agent):
        bad = {**GOOD_POSTING, "title": ""}
        result = agent.analyse(bad)
        assert "no title" in result

    def test_no_schema_posting_falls_through_to_llm_gracefully(self, agent):
        result = agent.analyse(NO_SCHEMA_POSTING)
        assert "no title" not in result
        assert "Risk Level" in result and "Action" in result


class TestResponseParsing:
    def test_parse_response_extracts_fields(self):
        response = "Risk Level: HIGH\nReason: Missing company info.\nAction: BLOCK"
        parsed = HrDetectorAgent.parse_response(response)
        assert parsed == {"risk_level": "HIGH", "action": "BLOCK", "reason": "Missing company info."}

    def test_is_fraud_true_for_medium_and_high(self):
        assert HrDetectorAgent.is_fraud({"risk_level": "MEDIUM"})
        assert HrDetectorAgent.is_fraud({"risk_level": "HIGH"})

    def test_is_fraud_false_for_low(self):
        assert not HrDetectorAgent.is_fraud({"risk_level": "LOW"})
