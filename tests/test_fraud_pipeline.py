# tests/test_fraud_pipeline.py
# Ported from fraud_detection/tests/test_agents.py — same assertions,
# import path moved under domains.bfsi_fraud. Requires OPENAI_API_KEY (the
# detector's LLM layer only fires for genuinely borderline transactions,
# but the client is constructed eagerly in __init__).

from domains.bfsi_fraud.agents.detector_agent import FraudDetectorAgent


class TestFraudDetectorAgent:

    def setup_method(self):
        """Runs before every test - creates fresh agent"""
        self.agent = FraudDetectorAgent(name="TestAgent")

    def test_agent_initialises_correctly(self):
        """Agent should have correct name and zero cases"""
        assert self.agent.name == "TestAgent"
        assert self.agent.cases_reviewed == 0

    def test_cases_reviewed_increments(self):
        """cases_reviewed should increase after each analyse call"""
        transaction = {"Amount": 100.0, "hour": 14, "Time": 50000}
        self.agent.analyse(transaction)
        assert self.agent.cases_reviewed == 1

    def test_transaction_dict_structure(self):
        """Transaction dict must have required keys"""
        transaction = {"Amount": 100.0, "hour": 14, "Time": 50000}
        assert "Amount" in transaction
        assert "hour" in transaction
        assert "Time" in transaction

    def test_high_risk_detection(self):
        """Zero amount transaction should return HIGH risk"""
        transaction = {"Amount": 0.0, "hour": 2, "Time": 50000}
        result = self.agent.analyse(transaction)
        assert "HIGH" in result

    def test_low_risk_detection(self):
        """Normal transaction should return LOW risk"""
        transaction = {"Amount": 150.0, "hour": 14, "Time": 50000}
        result = self.agent.analyse(transaction)
        assert "LOW" in result or "MEDIUM" in result

    def test_response_format(self):
        """Response must contain Risk Level, Reason, Action"""
        transaction = {"Amount": 100.0, "hour": 14, "Time": 50000}
        result = self.agent.analyse(transaction)
        assert "Risk Level" in result
        assert "Reason" in result
        assert "Action" in result


class TestFraudDetectorAgentParsing:
    """Static-method tests — no API key needed, pure string parsing."""

    def test_parse_response_extracts_fields(self):
        response = "Risk Level: HIGH\nReason: Zero amount.\nAction: BLOCK"
        parsed = FraudDetectorAgent.parse_response(response)
        assert parsed == {"risk_level": "HIGH", "action": "BLOCK", "reason": "Zero amount."}

    def test_is_fraud_true_for_medium_and_high(self):
        assert FraudDetectorAgent.is_fraud({"risk_level": "MEDIUM"})
        assert FraudDetectorAgent.is_fraud({"risk_level": "HIGH"})

    def test_is_fraud_false_for_low(self):
        assert not FraudDetectorAgent.is_fraud({"risk_level": "LOW"})

    def test_is_fraud_fails_safe_on_unknown(self):
        assert FraudDetectorAgent.is_fraud({"risk_level": "UNKNOWN"})

    def test_rule_based_filter_zero_amount(self):
        result = FraudDetectorAgent.rule_based_filter({"Amount": 0.0, "hour": 10})
        assert result is not None
        assert "HIGH" in result

    def test_rule_based_filter_normal_transaction_returns_none(self):
        result = FraudDetectorAgent.rule_based_filter({"Amount": 150.0, "hour": 14})
        assert result is None
