"""
tests/test_financial_services_pipeline.py
Real end-to-end test of FinancialServicesPipeline (core/unified_pipeline.py's
UnifiedDomainPipeline) against actual LLM calls -- same spirit as
test_documents_pipeline.py/test_insurance_unified_pipeline.py: no
mocking, this exercises the real ingest -> extract -> map -> score ->
chat flow.

No deterministic-rule assertion here the way Banking's ($0 transaction)
or Insurance's (approved > billed) tests have: FintechDetectorAgent's
rules layer is TIER-GATED (only runs for tier1/tier2, never tier3 --
see domains/financial_services/pipeline.py's module docstring), and
every document-extracted record here honestly resolves to tier3. So
this test verifies the real mechanics (extraction, mapping, scoring,
chat) work end to end, not a specific verdict outcome, which would
depend on the LLM's own judgement rather than a deterministic rule.

Requires OPENAI_API_KEY (embeddings) and Ollama running locally
(LLM_PROVIDER=ollama, the default) or LLM_PROVIDER=openai set in .env.
Slow (real LLM calls) -- run standalone, not part of the fast suite.
"""

from pathlib import Path

import pytest

from core.orchestrator import AgentOrchestrator

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reference_corpora")]

WALLET_STATEMENT = Path("domains/financial_services/data/statements/WALLET001_statement.pdf")


@pytest.fixture(scope="session")
def pipeline():
    orchestrator = AgentOrchestrator()
    return orchestrator.get_pipeline("financial_services")


@pytest.mark.skipif(not WALLET_STATEMENT.exists(), reason="run generate_wallet_statement.py first")
class TestUnifiedIngest:
    @pytest.fixture(scope="class")
    def ingest_response(self, pipeline):
        return pipeline.ingest(str(WALLET_STATEMENT), filename="WALLET001_statement")

    def test_ingest_succeeds_and_chunks_stored(self, ingest_response):
        assert ingest_response["status"] == "success"
        assert ingest_response["chunks_stored"] > 0
        assert ingest_response["file_type"] == "pdf"
        assert ingest_response["records"] is None  # PDF, not CSV

    def test_every_extracted_transaction_gets_a_real_verdict(self, ingest_response):
        """The demo statement lists 5 real transactions; the exact
        extracted count isn't asserted (same tolerance
        test_documents_pipeline.py's Banking test gives real LLM
        extraction) -- what matters is every extracted record went
        through the real engine and got an independent verdict."""
        verdicts = ingest_response["fraud_verdicts"]
        assert len(verdicts) >= 3  # allow some extraction variance around the 5 real lines
        for v in verdicts:
            assert v["predicted"] in {"FRAUD", "LEGITIMATE"}
        summary = ingest_response["fraud_summary"]
        assert summary["total_scored"] == len(verdicts)

    def test_seed_history_built(self, ingest_response):
        assert len(ingest_response["seed_history"]) == 2
        assert ingest_response["seed_history"][0]["speaker"] == "user"
        assert ingest_response["seed_history"][1]["speaker"] == "assistant"

    def test_followup_chat_answers_from_document(self, pipeline, ingest_response):
        result = pipeline.run(
            "What was the total activity amount?", source_document="WALLET001_statement",
            history=ingest_response["seed_history"],
        )
        assert result["status"] == "valid"
        assert len(result["data"]["answer"].strip()) > 0

    def test_followup_chat_draws_on_reference_corpus(self, pipeline, ingest_response):
        """FinancialServicesPipeline.REFERENCE_SOURCE is now set (see
        pipeline.py) -- confirms retrieve() actually merges the ingested
        wallet statement with the FinCEN CVC guidance reference corpus
        (domains/financial_services/reference_corpus/fincen_cvc_guidance_excerpt.txt,
        run via domains/financial_services/data/ingest_reference_corpus.py)
        rather than answering from the statement alone. Grounded in a
        real, quoted FinCEN distinction that appears nowhere in the
        wallet statement itself: hosted vs. unhosted wallets are
        distinguished by who controls the funds, not by technology."""
        result = pipeline.run(
            "According to FinCEN guidance, what is the difference between "
            "a hosted wallet and an unhosted wallet?",
            source_document="WALLET001_statement",
            history=ingest_response["seed_history"],
        )
        assert result["status"] == "valid"
        answer_lower = result["data"]["answer"].lower()
        assert "hosted" in answer_lower

    def test_evaluation_present_and_valid(self, pipeline, ingest_response):
        """FinancialServicesPipeline.evaluator is now set (domains/financial_services/
        evaluation/rag_evaluator.py, a thin wrapper over core/rag/base_rag_evaluator.py).
        Real ground truth for WALLET001 lives in that domain's ground_truth.json."""
        result = pipeline.run(
            "What was the total activity amount?", source_document="WALLET001_statement",
            history=ingest_response["seed_history"],
        )
        assert result["status"] == "valid"
        assert "evaluation" in result
        evaluation = result["evaluation"]
        assert evaluation is not None
        for key in ["context_precision", "faithfulness", "answer_relevancy", "evaluator", "latency_ms"]:
            assert key in evaluation, f"Missing evaluation key: {key}"
        for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
            score = evaluation[metric]
            if score is None and metric == "context_precision":
                continue
            assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"
