"""
tests/test_documents_pipeline.py
Adapted from document_intelligence_system/tests/test_api.py. The original
suite drove a running FastAPI server over HTTP; Prism has no separate API
service (single Streamlit deployment), so these call BankingPipeline
(domains/banking/pipeline.py, unifying fraud detection and document chat
under one pipeline -- see core/unified_pipeline.py) directly in-process
instead. Same underlying assertions on ingest/analyze behaviour as the
original bfsi_documents suite; the SSE-streaming test class is dropped
since there's no streaming endpoint to test.

Requires OPENAI_API_KEY (embeddings always go through OpenAI) and, for the
default LLM_PROVIDER=ollama, a running Ollama with llama3.1:8b pulled.
Session-scoped fixtures keep this to 1 ingest + 1 analyze call for the
whole suite, but each chains multiple sequential LLM round-trips (HyDE +
query variants + reasoning + RAG eval), so on CPU-only Ollama this can take
several minutes — same performance profile as the original suite.

Run with:
    pytest tests/test_documents_pipeline.py -v
"""

import pytest
from pathlib import Path
from fpdf import FPDF

from core.orchestrator import AgentOrchestrator


# ══════════════════════════════════════════════════════════════════════════════
# SESSION-SCOPED FIXTURES  (LLM called once per session)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def pipeline():
    orchestrator = AgentOrchestrator()
    return orchestrator.get_pipeline("banking")


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    """Generate a minimal bank statement PDF for testing."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "test_bank_statement.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "HDFC Bank Statement - Test 2024", ln=True)
    pdf.cell(0, 10, "Account: ACC999999", ln=True)
    pdf.cell(0, 10, "Customer: Test User", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Date         Description              Amount    Balance", ln=True)
    pdf.cell(0, 10, "01-Jan-2024  Salary credit            +50000    50000", ln=True)
    pdf.cell(0, 10, "05-Jan-2024  ATM withdrawal           -20000    30000", ln=True)
    pdf.cell(0, 10, "10-Jan-2024  UPI to Amazon            -2000     28000", ln=True)
    pdf.cell(0, 10, "15-Jan-2024  UPI received Rahul       +10000    38000", ln=True)
    pdf.cell(0, 10, "20-Jan-2024  ATM withdrawal           -15000    23000", ln=True)
    pdf.cell(0, 10, "31-Jan-2024  Closing balance                    23000", ln=True)
    pdf.output(str(pdf_path))
    return pdf_path


@pytest.fixture(scope="session")
def ingest_response(pipeline, sample_pdf) -> dict:
    """Ingest the sample PDF once and return the full response dict."""
    return pipeline.ingest(str(sample_pdf), filename=sample_pdf.stem)


@pytest.fixture(scope="session")
def ingested_doc(ingest_response) -> str:
    """Return the ingested document name."""
    return ingest_response["document"]


@pytest.fixture(scope="session")
def analyze_response(pipeline, ingested_doc) -> dict:
    """
    Call pipeline.run() once and cache the response for the entire session.
    All TestAnalyze tests share this single LLM call.
    """
    return pipeline.run("What are the large withdrawals?", source_document=ingested_doc)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineConstruction:
    def test_pipeline_builds_all_agents(self, pipeline):
        assert pipeline.loader is not None
        assert pipeline.retriever is not None
        assert pipeline.reasoning is not None
        assert pipeline.validator is not None
        assert pipeline.evaluator is not None


# ══════════════════════════════════════════════════════════════════════════════
# INGEST
# ══════════════════════════════════════════════════════════════════════════════


class TestIngest:
    def test_ingest_returns_success_status(self, ingest_response):
        assert ingest_response["status"] == "success"

    def test_ingest_returns_document_name(self, ingest_response, sample_pdf):
        assert "document" in ingest_response
        assert ingest_response["document"] == sample_pdf.stem

    def test_ingest_returns_chunks_stored(self, ingest_response):
        assert "chunks_stored" in ingest_response
        assert ingest_response["chunks_stored"] >= 1

    def test_ingest_no_temp_filename_leak(self, ingest_response):
        """Verify the temp path is not returned as document name."""
        assert not ingest_response["document"].startswith("tmp")

    def test_ingest_scores_every_extracted_transaction(self, ingest_response):
        """ingest() now routes each extracted transaction through the
        SAME rules->tiered ML->LLM engine the CSV fraud path uses --
        core/unified_pipeline.py's UnifiedDomainPipeline, replacing the
        old separate LLM-only anomaly_scan capability. The sample
        statement has 5 real transaction lines; the exact extracted count
        isn't asserted (this file's own TestAnalyze.test_analyze_transaction_fields
        doesn't either) -- real LLM extraction can occasionally include or
        drop a borderline line (e.g. a closing-balance row), same
        tolerance the rest of this suite already gives it. What matters is
        that every extracted record got a real, independent verdict."""
        assert "fraud_verdicts" in ingest_response
        verdicts = ingest_response["fraud_verdicts"]
        assert len(verdicts) >= 4  # allow +/-1 extraction variance around the 5 real lines
        for v in verdicts:
            assert v["predicted"] in {"FRAUD", "LEGITIMATE"}
        summary = ingest_response["fraud_summary"]
        assert summary["total_scored"] == len(verdicts)
        assert summary["flagged"] + summary["clean"] == len(verdicts)

    def test_ingest_builds_seed_history(self, ingest_response):
        assert len(ingest_response["seed_history"]) == 2
        assert ingest_response["seed_history"][0]["speaker"] == "user"
        assert ingest_response["seed_history"][1]["speaker"] == "assistant"


# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE  (all tests share one cached LLM response)
# ══════════════════════════════════════════════════════════════════════════════


class TestAnalyze:
    def test_analyze_status_is_valid(self, analyze_response):
        assert analyze_response["status"] == "valid"

    def test_analyze_data_has_required_fields(self, analyze_response):
        data = analyze_response["data"]
        for field in [
            "answer",
            "transactions",
            "total_amount",
            "confidence",
            "source_document",
            "anomaly_flag",
        ]:
            assert field in data, f"Missing field: {field}"

    def test_analyze_answer_is_non_empty_string(self, analyze_response):
        answer = analyze_response["data"]["answer"]
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_analyze_transactions_is_list(self, analyze_response):
        assert isinstance(analyze_response["data"]["transactions"], list)

    def test_analyze_transaction_fields(self, analyze_response):
        for txn in analyze_response["data"]["transactions"]:
            assert "date" in txn
            assert "amount" in txn
            assert "type" in txn
            assert "description" in txn
            assert isinstance(txn["amount"], (int, float))

    def test_analyze_confidence_in_range(self, analyze_response):
        conf = analyze_response["data"]["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_analyze_total_amount_is_numeric(self, analyze_response):
        total = analyze_response["data"]["total_amount"]
        assert isinstance(total, (int, float))

    def test_analyze_anomaly_flag_is_bool(self, analyze_response):
        assert isinstance(analyze_response["data"]["anomaly_flag"], bool)

    def test_analyze_includes_evaluation_key(self, analyze_response):
        assert "evaluation" in analyze_response

    def test_analyze_evaluation_has_required_metrics(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            for key in [
                "context_precision",
                "faithfulness",
                "answer_relevancy",
                "evaluator",
                "latency_ms",
            ]:
                assert key in evaluation, f"Missing evaluation key: {key}"

    def test_analyze_evaluation_scores_in_range(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
                score = evaluation[metric]
                # context_precision is None under RAGASEvaluator (LLM_PROVIDER=openai)
                # when the query doesn't match a known ground-truth question --
                # "not measurable", not a range violation. CustomRAGEvaluator
                # (LLM_PROVIDER=ollama, what these tests run against by default)
                # never returns None here, but don't assume that backend forever.
                if score is None and metric == "context_precision":
                    continue
                assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

    def test_analyze_evaluation_evaluator_is_string(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            assert isinstance(evaluation["evaluator"], str)
            assert len(evaluation["evaluator"]) > 0

    def test_analyze_evaluation_latency_is_positive(self, analyze_response):
        evaluation = analyze_response.get("evaluation")
        if evaluation is not None:
            assert isinstance(evaluation["latency_ms"], int)
            assert evaluation["latency_ms"] >= 0


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_analyze_with_null_source_document_does_not_raise(self, pipeline, ingested_doc):
        """source_document is optional — None should not raise, even though
        the answer quality may be worse without it scoping retrieval."""
        result = pipeline.run("What are the withdrawals?", source_document=None)
        assert result["status"] in {"valid", "invalid"}


class TestReferenceCorpus:
    def test_followup_query_draws_on_reference_corpus(self, pipeline, ingested_doc):
        """BankingPipeline.REFERENCE_SOURCE is set (see pipeline.py) --
        confirms retrieve() actually merges the ingested statement with
        the Regulation E reference corpus
        (domains/banking/reference_corpus/regulation_e_excerpt.txt, run
        via domains/banking/documents/data/ingest_reference_corpus.py)
        rather than answering from the statement alone. Grounded in a
        real, quoted Regulation E 1005.6 figure that appears nowhere in
        the bank statement itself: prompt notice within 2 business days
        caps consumer liability at $50."""
        result = pipeline.run(
            "Under Regulation E, if a consumer notifies their bank within "
            "2 business days of losing their debit card, what is the most "
            "they can be held liable for?",
            source_document=ingested_doc,
        )
        assert result["status"] == "valid"
        assert "50" in result["data"]["answer"]
