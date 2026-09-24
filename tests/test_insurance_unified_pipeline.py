"""
tests/test_insurance_unified_pipeline.py
Real end-to-end test of InsurancePipeline (core/unified_pipeline.py's
UnifiedDomainPipeline) against actual LLM calls -- same spirit as
test_documents_pipeline.py/test_payroll_pipeline.py: no mocking, this
exercises the real ingest -> extract -> map -> score -> chat flow.

Requires OPENAI_API_KEY in .env (embeddings) and Ollama running locally
(LLM_PROVIDER=ollama, the default) or LLM_PROVIDER=openai set in .env.
Slow (real LLM calls) -- run standalone, not part of the fast suite.
"""

from pathlib import Path

import pytest

from core.orchestrator import AgentOrchestrator

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reference_corpora")]

CLM001 = Path("domains/insurance/data/statements/CLM001_claim.pdf")   # clean
CLM003 = Path("domains/insurance/data/statements/CLM003_claim.pdf")   # approved > billed


@pytest.fixture(scope="session")
def pipeline():
    orchestrator = AgentOrchestrator()
    return orchestrator.get_pipeline("insurance")


@pytest.mark.skipif(not CLM001.exists(), reason="run generate_claim_document.py first")
class TestUnifiedIngestCleanClaim:
    @pytest.fixture(scope="class")
    def ingest_response(self, pipeline):
        return pipeline.ingest(str(CLM001), filename="CLM001_claim")

    def test_ingest_succeeds_and_chunks_stored(self, ingest_response):
        assert ingest_response["status"] == "success"
        assert ingest_response["chunks_stored"] > 0
        assert ingest_response["file_type"] == "pdf"
        assert ingest_response["records"] is None  # PDF, not CSV

    def test_auto_fraud_verdict_present_and_clean(self, ingest_response):
        # A claim/EOB represents exactly one artifact -> at most one
        # verdict in the list (see core/unified_pipeline.py's docstring
        # for why this differs from Banking's many-transactions-per-
        # statement case).
        verdicts = ingest_response["fraud_verdicts"]
        assert len(verdicts) == 1
        assert verdicts[0]["predicted"] in {"FRAUD", "LEGITIMATE"}
        assert ingest_response["fraud_summary"]["total_scored"] == 1

    def test_seed_history_built(self, ingest_response):
        assert len(ingest_response["seed_history"]) == 2
        assert ingest_response["seed_history"][0]["speaker"] == "user"
        assert ingest_response["seed_history"][1]["speaker"] == "assistant"

    def test_followup_chat_answers_from_document_and_reference_corpus(self, pipeline, ingest_response):
        # Grounded in the uploaded document (the claim amount) --
        # confirms retrieval + reasoning actually work end to end.
        result = pipeline.run(
            "What was the billed claim amount?", source_document="CLM001_claim",
            history=ingest_response["seed_history"],
        )
        assert result["status"] == "valid"
        assert "480" in result["data"]["answer"]
        self._assert_valid_evaluation(result)

    @staticmethod
    def _assert_valid_evaluation(result):
        """InsurancePipeline.evaluator is now set (domains/insurance/evaluation/
        rag_evaluator.py, a thin wrapper over core/rag/base_rag_evaluator.py --
        the same evaluation logic Banking's already-production evaluator uses,
        generalized rather than copy-pasted). Real ground truth for CLM001
        lives in domains/insurance/evaluation/ground_truth.json."""
        assert "evaluation" in result
        evaluation = result["evaluation"]
        assert evaluation is not None
        for key in ["context_precision", "faithfulness", "answer_relevancy", "evaluator", "latency_ms"]:
            assert key in evaluation, f"Missing evaluation key: {key}"
        for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
            score = evaluation[metric]
            if score is None and metric == "context_precision":
                continue  # RAGASEvaluator only, when no ground-truth question matched
            assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"
        assert isinstance(evaluation["evaluator"], str) and evaluation["evaluator"]


@pytest.mark.skipif(not CLM003.exists(), reason="run generate_claim_document.py first")
class TestUnifiedIngestRuleViolatingClaim:
    @pytest.fixture(scope="class")
    def ingest_response(self, pipeline):
        return pipeline.ingest(str(CLM003), filename="CLM003_claim")

    def test_rule_violation_flagged_as_fraud(self, ingest_response):
        # CLM003: approved ($9,999) > billed ($500) -- a real rules-layer
        # trigger in HealthcareDetectorAgent's _rule_check_tier1, reached
        # via the SAME detect_tier()-routed engine the CSV path uses,
        # once the PDF's extracted fields are mapped to Claim_Amount/
        # Approved_Amount/Days_Between_Service_and_Claim.
        verdicts = ingest_response["fraud_verdicts"]
        assert len(verdicts) == 1
        assert verdicts[0]["predicted"] == "FRAUD"
        assert "exceeds the claimed amount" in verdicts[0]["detector_response"]
