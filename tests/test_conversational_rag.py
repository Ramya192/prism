"""
tests/test_conversational_rag.py
Tests for the conversation-history threading added to banking documents' and
payroll's ReasoningAgent, and the shared seed_history builder that turns
an auto-scan verdict into the first conversation turn. All pure string/
dict construction -- no real LLM call happens in any of these (a dummy
OPENAI_API_KEY is enough to construct the agent; ReasoningAgent's
__init__ builds an LLM client object but never invokes it here).
"""

from domains.banking.documents.agents.reasoning_agent import ReasoningAgent as BfsiReasoningAgent
from domains.payroll_hr.payroll.agents.reasoning_agent import ReasoningAgent as PayrollReasoningAgent
from core.rag.seed_history import build_seed_history


class TestRenderHistory:
    def test_empty_history_renders_nothing(self):
        assert BfsiReasoningAgent._render_history(None) == ""
        assert BfsiReasoningAgent._render_history([]) == ""

    def test_renders_user_and_assistant_turns(self):
        history = [
            {"speaker": "user", "text": "What was the net pay?"},
            {"speaker": "assistant", "text": "Net pay was $1,967.53."},
        ]
        rendered = PayrollReasoningAgent._render_history(history)
        assert "User: What was the net pay?" in rendered
        assert "You: Net pay was $1,967.53." in rendered
        assert "Previous conversation" in rendered

    def test_history_appears_in_built_prompt(self):
        agent = PayrollReasoningAgent()
        history = [{"speaker": "assistant", "text": "Flagged: reconciliation mismatch."}]
        prompt = agent._build_prompt("why was that flagged?", [], "test_doc", history)
        assert "Flagged: reconciliation mismatch." in prompt
        assert "why was that flagged?" in prompt

    def test_no_history_produces_no_previous_conversation_section(self):
        agent = BfsiReasoningAgent()
        prompt = agent._build_prompt("what is the balance?", [], "test_doc", None)
        assert "Previous conversation" not in prompt


class TestBuildSeedHistory:
    def test_invalid_scan_returns_empty(self):
        assert build_seed_history("check this", {"status": "invalid", "errors": ["x"]}) == []

    def test_flagged_scan_renders_warning_in_seed(self):
        scan = {"status": "valid", "data": {"answer": "Gross pay was $3,010.", "flag": True, "flag_reason": "mismatch"}}
        seed = build_seed_history("audit this payslip", scan)
        assert len(seed) == 2
        assert seed[0] == {"speaker": "user", "text": "audit this payslip"}
        assert seed[1]["speaker"] == "assistant"
        assert "Gross pay was $3,010." in seed[1]["text"]
        assert "⚠ Flagged: mismatch" in seed[1]["text"]

    def test_clean_scan_renders_no_anomalies_message(self):
        scan = {"status": "valid", "data": {"answer": "Everything checks out.", "flag": False, "flag_reason": None}}
        seed = build_seed_history("audit this payslip", scan)
        assert "No anomalies found" in seed[1]["text"]
        assert "⚠" not in seed[1]["text"]

    def test_nothing_scored_does_not_claim_a_clean_check(self):
        """No verdict at all (not this domain's kind of document) must not read
        as 'no anomalies found' -- that would imply a check that never ran."""
        scan = {"status": "valid", "data": {"answer": "An EOB.", "flag": False,
                                            "flag_reason": "No fraud check was run on this document."}}
        seed = build_seed_history("what is this", scan)
        assert "No fraud check was run" in seed[1]["text"]
        assert "No anomalies found" not in seed[1]["text"]

    def test_banking_documents_field_names_also_supported(self):
        """banking documents uses anomaly_flag/anomaly_reason instead of
        flag/flag_reason -- the shared builder must handle both."""
        scan = {"status": "valid", "data": {"answer": "3 transactions found.", "anomaly_flag": True, "anomaly_reason": "duplicate charge"}}
        seed = build_seed_history("scan for anomalies", scan)
        assert "⚠ Flagged: duplicate charge" in seed[1]["text"]


class TestDocumentTaggedSeed:
    def test_seed_turns_name_their_document(self):
        scan = {"status": "valid", "data": {"answer": "5 transactions.", "flag": False, "flag_reason": None}}
        seed = build_seed_history("scan", scan, document="march.pdf")
        assert seed[0]["text"].startswith("[march.pdf] ")
        assert seed[1]["text"].startswith("[march.pdf] ")
        assert "No anomalies found" in seed[1]["text"]

    def test_untagged_seed_is_unchanged(self):
        scan = {"status": "valid", "data": {"answer": "x", "flag": False, "flag_reason": None}}
        assert build_seed_history("scan", scan)[0] == {"speaker": "user", "text": "scan"}


class TestPayrollPolicyQuestions:
    def test_prompt_routes_policy_questions_to_reference_corpus(self):
        from core.rag.reference_corpus import REFERENCE_SOURCE_ID
        prompt = PayrollReasoningAgent()._build_prompt("how are bonuses withheld?", [], "slip.pdf", None)
        assert "policy, regulation or rules question" in prompt
        assert f'"{REFERENCE_SOURCE_ID}"' in prompt
