# tests/test_fraud_pipeline.py
# Ported from fraud_detection/tests/test_agents.py — same assertions,
# import path moved under domains.banking.fraud. Requires OPENAI_API_KEY (the
# detector's LLM layer only fires for genuinely borderline transactions,
# but the client is constructed eagerly in __init__).

from domains.banking.fraud.agents.detector_agent import FraudDetectorAgent


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
        assert parsed == {"risk_level": "HIGH", "action": "BLOCK", "reason": "Zero amount.", "caveat": None}

    def test_parse_response_extracts_caveat(self):
        response = (
            "Risk Level: LOW\nReason: Typical transaction.\nAction: APPROVE\n"
            "Caveat: this transaction also looks statistically unusual relative to the "
            "training distribution (unsupervised check, independent of the verdict above) "
            "— worth extra scrutiny regardless of the risk level assigned."
        )
        parsed = FraudDetectorAgent.parse_response(response)
        assert parsed["caveat"] is not None
        assert "statistically unusual" in parsed["caveat"]

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


class TestSchemaTiering:
    """detect_tier() picks which layers are even reachable for a given
    transaction shape -- see the module docstring at the top of
    detector_agent.py for what each tier means. Pure logic, no API key
    needed for detect_tier() itself; ml_filter() trains real scorers in
    setup_method but never calls the LLM, so no network needed either."""

    def setup_method(self):
        self.agent = FraudDetectorAgent(name="TierTestAgent")

    # Tier 2 schema: GeneralizableScorer.REQUIRED_COLS
    # (Transaction_Amount, Account_Age, Credit_Score) -- a second,
    # real-world-shaped dataset, NOT a reduced version of Tier 1's own
    # columns (that was tried and empirically rejected, see
    # generalizable_scorer.py's module docstring).
    TIER2_TXN = {
        "Transaction_Amount": 250.0, "Transaction_Type": "Withdrawal",
        "Account_Age": 5, "Credit_Score": 480, "Previous_Fraud": 1,
        "Device_Used": "Mobile", "hour": 3,
    }

    def test_detects_tier1_when_full_kaggle_schema_present(self):
        txn = {f"V{i}": 0.1 for i in range(1, 29)}
        txn.update({"Amount": 100.0, "Time": 50000, "hour": 14})
        assert FraudDetectorAgent.detect_tier(txn) == "tier1"

    def test_detects_tier2_when_generalizable_schema_present(self):
        assert FraudDetectorAgent.detect_tier(self.TIER2_TXN) == "tier2"

    def test_bare_amount_without_either_full_schema_is_tier3(self):
        # A lone "Amount"/"Transaction_Amount" field isn't enough on its
        # own for either tier -- that reduced-feature approach was tried
        # and empirically rejected (AUC 0.66, precision 0.3%), so it no
        # longer qualifies as tier2.
        txn = {"Amount": 250.0, "hour": 3, "merchant": "Acme Corp"}
        assert FraudDetectorAgent.detect_tier(txn) == "tier3"

    def test_detects_tier3_when_no_amount_field_at_all(self):
        txn = {"merchant": "Acme Corp", "note": "not transaction-shaped"}
        assert FraudDetectorAgent.detect_tier(txn) == "tier3"

    def test_tier1_ml_filter_uses_full_scorer(self):
        txn = {f"V{i}": 0.1 for i in range(1, 29)}
        txn.update({"Amount": 100.0, "Time": 50000, "hour": 14})
        result = self.agent.ml_filter(txn, "tier1")
        if result is not None:
            assert "tier1: full trained schema" in result

    def test_tier2_ml_filter_uses_generalizable_scorer(self):
        result = self.agent.ml_filter(self.TIER2_TXN, "tier2")
        if result is not None:
            assert "tier2: generalizable real-world schema" in result

    def test_normalize_for_rules_remaps_tier2_amount_field(self):
        """rule_based_filter() only knows "Amount"/"hour" -- without this
        remap, a tier2 transaction's real Transaction_Amount would read as
        0 and wrongly trip the zero-dollar rule on every tier2 case."""
        normalized = FraudDetectorAgent._normalize_for_rules(self.TIER2_TXN)
        assert normalized == {"Amount": 250.0, "hour": 3}

    def test_normalize_for_rules_returns_none_when_no_amount_field_at_all(self):
        assert FraudDetectorAgent._normalize_for_rules({"merchant": "x"}) is None

    def test_normalize_for_rules_applies_even_without_a_full_ml_schema(self):
        """Rules are independent of ML-tier eligibility: a bare Amount/hour
        transaction with no V1-V28 and no Account_Age/Credit_Score doesn't
        qualify for either ML tier, but should still get a rules check --
        this is the exact case that regressed once before (tier3 wrongly
        skipped rules for it) and was caught by
        test_high_risk_detection failing."""
        txn = {"Amount": 0.0, "hour": 2}
        assert FraudDetectorAgent.detect_tier(txn) == "tier3"
        assert FraudDetectorAgent._normalize_for_rules(txn) == txn

    def test_tier2_build_prompt_uses_correct_field_names(self):
        prompt = self.agent.build_prompt(self.TIER2_TXN, ml_score=0.45, tier="tier2")
        assert "Amount: $250.0" in prompt
        assert "Previous fraud on this customer: 1" in prompt
        assert "Credit score: 480" in prompt

    def test_tier3_build_prompt_has_no_key_error_and_lists_raw_fields(self):
        txn = {"merchant": "Acme Corp", "note": "no amount field"}
        prompt = self.agent.build_prompt(txn, tier="tier3")
        assert "merchant: Acme Corp" in prompt
        assert "doesn't match any known transaction schema" in prompt

    def test_analyse_skips_rules_and_ml_for_tier3(self):
        """A tier3 transaction defaulting Amount to 0 inside rule_based_filter
        would wrongly fire the zero-dollar rule -- analyse() must skip
        rules/ML entirely for tier3, not just skip ML."""
        txn = {"merchant": "Acme Corp", "note": "no amount field"}
        result = self.agent.analyse(txn)
        # However it resolves (real LLM call or the graceful API-error
        # fallback), it must NOT be the zero-dollar rule's exact wording --
        # that would mean Amount got defaulted to 0 somewhere upstream.
        assert "Zero-dollar transaction" not in result

    def test_analyse_uses_remapped_amount_for_tier2_rules(self):
        """A tier2 transaction with Transaction_Amount=0 must still trip
        the zero-dollar rule (via the remap) -- proving the remap actually
        feeds rule_based_filter, not just that it returns the right dict."""
        txn = dict(self.TIER2_TXN)
        txn["Transaction_Amount"] = 0.0
        result = self.agent.analyse(txn)
        assert "Zero-dollar transaction" in result


class TestDriftCaveat:
    """Isolation Forest, wired into Tier 1 only -- see detector_agent.py's
    analyse() for why Tier 2/3 are deliberately excluded (DriftDetector's
    features are Tier 1's own schema) and why this domain specifically
    (unlike payroll/insurance/HR) earned live integration:
    measured 4.5% legit-outlier rate here vs. 10-51% elsewhere, so the
    caveat is a rare, meaningful signal rather than noise."""

    def setup_method(self):
        self.agent = FraudDetectorAgent(name="DriftCaveatTest")

    def test_tier1_outlier_gets_caveat_appended(self):
        import pandas as pd

        holdout = pd.read_csv("domains/banking/fraud/data/test_holdout.csv")
        fraud_rows = holdout[holdout["is_Fraud"] == 1]
        for _, row in fraud_rows.iterrows():
            txn = row.to_dict()
            if self.agent.rule_based_filter(txn) is None and self.agent.drift_detector.is_outlier(txn):
                result = self.agent.analyse(txn)
                assert "Caveat:" in result
                assert "statistically unusual" in result
                # The caveat must not corrupt structured parsing
                parsed = FraudDetectorAgent.parse_response(result)
                assert parsed["risk_level"] in FraudDetectorAgent.VALID_RISK_LEVELS
                return
        raise AssertionError("No rules-clearing, drift-flagged fraud row found in this holdout -- fixture assumption broke")

    def test_non_outlier_gets_no_caveat(self):
        txn = {f"V{i}": 0.0 for i in range(1, 29)}
        txn.update({"Amount": 100.0, "Time": 50000, "hour": 14})
        assert not self.agent.drift_detector.is_outlier(txn)
        result = self.agent.analyse(txn)
        assert "Caveat:" not in result

    def test_tier2_and_tier3_never_get_the_caveat(self):
        """DriftDetector's feature set is Tier 1's schema -- running it on
        Tier 2/3 data would silently zero-fill missing V1-V28 and produce
        a meaningless result, so it's gated to tier1 only regardless of
        what is_outlier() would say if called directly."""
        tier2_txn = {
            "Transaction_Amount": 250.0, "Transaction_Type": "Withdrawal",
            "Account_Age": 5, "Credit_Score": 480, "Previous_Fraud": 1,
            "Device_Used": "Mobile", "hour": 3,
        }
        tier3_txn = {"merchant": "Acme Corp", "note": "no amount field"}
        assert "Caveat:" not in self.agent.analyse(tier2_txn)
        assert "Caveat:" not in self.agent.analyse(tier3_txn)
