"""
tests/test_payroll_hr_handbook_pipeline.py
Real end-to-end test of PayrollHrPipeline's document path against an HR
EMPLOYEE HANDBOOK -- not a payslip. This is the genuinely different HR
document type payroll_hr's document/RAG side didn't have a test for yet.

The FIRST version of this test suite assumed a handbook is not a fraud-
scoreable artifact and that core/unified_pipeline.py's base ingest()
already degraded gracefully for it (fraud_verdicts: [] because
_map_extraction_to_records() would find no gross_pay/net_pay fields).
Running it for real against the actual SHRM handbook below proved that
assumption WRONG: asked EXTRACTION_QUERY's payslip-specific question, the
LLM didn't return nulls -- it fabricated a plausible-looking
GROSS/Deduction/Net_Pay record with dollar figures that appear NOWHERE in
the real handbook text (verified directly against the source), which got
scored as a genuine FRAUD verdict on a document that was never a payslip.
The real fix -- PayrollHrPipeline._looks_like_payslip()/_extract_document()
(see pipeline.py's class docstring) -- gates the payslip extraction query
on a cheap content classification BEFORE it ever runs, so a non-payslip
document never reaches a prompt that pressures the LLM into inventing
payslip-shaped numbers at all. This suite verifies THAT fix, not the base
class's generic (and, for this document type, insufficient) empty-list
behavior.

Requires domains/payroll_hr/data/handbooks/SHRM_Sample_Employee_Handbook_2023.docx
locally (see that directory's README.md for why it isn't committed) --
skips entirely if absent. Also requires OPENAI_API_KEY (embeddings) and
Ollama running locally (LLM_PROVIDER=ollama, the default) or
LLM_PROVIDER=openai set in .env.

Run with:
    pytest tests/test_payroll_hr_handbook_pipeline.py -v
"""

from pathlib import Path

import pytest

from core.orchestrator import AgentOrchestrator
from domains.payroll_hr.pipeline import HANDBOOK_QUERY

HANDBOOK = Path("domains/payroll_hr/data/handbooks/SHRM_Sample_Employee_Handbook_2023.docx")


@pytest.fixture(scope="session")
def pipeline():
    orchestrator = AgentOrchestrator()
    return orchestrator.get_pipeline("payroll_hr")


@pytest.mark.skipif(not HANDBOOK.exists(), reason="see domains/payroll_hr/data/handbooks/README.md")
class TestHandbookIngest:
    @pytest.fixture(scope="class")
    def ingest_response(self, pipeline):
        return pipeline.ingest(str(HANDBOOK), filename=HANDBOOK.stem)

    def test_ingest_succeeds_and_chunks_stored(self, ingest_response):
        assert ingest_response["status"] == "success"
        assert ingest_response["chunks_stored"] > 0
        assert ingest_response["file_type"] == "docx"
        assert ingest_response["records"] is None  # DOCX, not CSV

    def test_no_forced_fraud_verdict(self, ingest_response):
        """A real, confirmed bug (not a hypothetical) is what this
        guards: an earlier version of PayrollHrPipeline trusted
        _map_extraction_to_records() to honestly return [] for a
        non-payslip document, on the theory that the LLM would return
        null gross/net pay fields when asked EXTRACTION_QUERY's payslip-
        specific question against a handbook. It didn't -- it fabricated
        a plausible-looking GROSS/Deduction/Net_Pay record with dollar
        figures that appear NOWHERE in the real handbook text, and that
        got scored as a real FRAUD verdict. The actual fix
        (PayrollHrPipeline._looks_like_payslip(), gating
        _extract_document() before EXTRACTION_QUERY ever runs) is what
        this test verifies -- not the base class's generic empty-list
        behavior, which this document type proved insufficient on its
        own."""
        assert ingest_response["fraud_verdicts"] == []
        assert ingest_response["fraud_summary"] == {"total_scored": 0, "flagged": 0, "clean": 0}

    def test_seed_history_gives_a_real_summary(self, ingest_response):
        """With the classification gate, the seed history's opening
        question is HANDBOOK_QUERY (not the payslip-specific
        EXTRACTION_QUERY -- see core/unified_pipeline.py's
        _extract_document() docstring for why query_used is threaded
        through), and the assistant turn is a real plain-text summary of
        the handbook, not a fraud-verdict message at all."""
        seed = ingest_response["seed_history"]
        assert len(seed) == 2
        assert seed[0]["text"] == HANDBOOK_QUERY
        assert len(seed[1]["text"].strip()) > 0

    def test_followup_chat_answers_from_real_handbook_content(self, pipeline, ingest_response):
        """Grounded in a real, quoted figure from the actual fetched SHRM
        template's Overtime section (not a placeholder field): "Nonexempt
        employees will be paid overtime compensation at the rate of one
        and one-half their regular rate of pay for all hours over 40
        actually worked in a single workweek." Confirms retrieval +
        reasoning work end to end against a non-payslip document type,
        using the exact same RAG agents as the payslip path."""
        doc = ingest_response["document"]
        result = pipeline.run(
            "According to this employee handbook, what is the overtime pay rate "
            "for nonexempt employees and after how many hours in a workweek does "
            "it apply?",
            source_document=doc,
        )
        assert result["status"] == "valid"
        answer = result["data"]["answer"]
        assert "40" in answer
        assert "1.5" in answer or "one and one-half" in answer.lower() or "one-half" in answer.lower()
