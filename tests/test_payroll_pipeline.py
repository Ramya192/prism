"""
tests/test_payroll_pipeline.py
Mirrors tests/test_documents_pipeline.py's structure for the payroll_hr
domain (PayrollHrPipeline, unifying fraud detection and payslip document
chat -- see core/unified_pipeline.py). Uses
domains/payroll_hr/payroll/tools/generate_payroll_register.py's own
compute_payslip()/generate_pdf() to build a correct payslip and a
deliberately-miscalculated one, so this suite doesn't depend on demo PDFs
having been generated on disk first.

Requires OPENAI_API_KEY (embeddings always go through OpenAI) and, for the
default LLM_PROVIDER=ollama, a running Ollama with llama3.1:8b pulled --
same requirement as test_documents_pipeline.py.

Run with:
    pytest tests/test_payroll_pipeline.py -v
"""

import pytest
from pathlib import Path

from core.orchestrator import AgentOrchestrator
from domains.payroll_hr.payroll.tools.generate_payroll_register import compute_payslip, generate_pdf


@pytest.fixture(scope="session")
def pipeline():
    orchestrator = AgentOrchestrator()
    return orchestrator.get_pipeline("payroll_hr")


@pytest.fixture(scope="session")
def correct_payslip_pdf(tmp_path_factory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("payslips")
    pdf_path = tmp_dir / "test_correct_payslip.pdf"
    payslip = compute_payslip(
        "EMPTEST1", "Test Employee", 80.0, 35.0, 4.0, break_reconciliation=False
    )
    generate_pdf(payslip, pdf_path, "March 1, 2026 — March 14, 2026", "March 20, 2026")
    return pdf_path


@pytest.fixture(scope="session")
def broken_payslip_pdf(tmp_path_factory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("payslips")
    pdf_path = tmp_dir / "test_broken_payslip.pdf"
    payslip = compute_payslip(
        "EMPTEST2", "Test Employee Two", 80.0, 38.0, 6.0, break_reconciliation=True
    )
    generate_pdf(payslip, pdf_path, "March 1, 2026 — March 14, 2026", "March 20, 2026")
    return pdf_path


@pytest.fixture(scope="session")
def correct_ingest_response(pipeline, correct_payslip_pdf) -> dict:
    return pipeline.ingest(str(correct_payslip_pdf), filename=correct_payslip_pdf.stem)


@pytest.fixture(scope="session")
def broken_ingest_response(pipeline, broken_payslip_pdf) -> dict:
    return pipeline.ingest(str(broken_payslip_pdf), filename=broken_payslip_pdf.stem)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineConstruction:
    def test_pipeline_builds_all_agents(self, pipeline):
        assert pipeline.loader is not None
        assert pipeline.retriever is not None
        assert pipeline.reasoning is not None
        assert pipeline.validator is not None


# ══════════════════════════════════════════════════════════════════════════════
# INGEST + AUTO-SCAN  (the one-upload verdict — no question needs to be asked)
# ══════════════════════════════════════════════════════════════════════════════


class TestIngestAndAutoScan:
    def test_ingest_returns_success_status(self, correct_ingest_response):
        assert correct_ingest_response["status"] == "success"

    def test_ingest_scores_the_payslip_through_the_real_engine(self, correct_ingest_response):
        """ingest() now maps the extracted line items into the SAME
        GROSS/Deduction/Net_Pay Tier 2 schema the CSV fraud path uses
        (core/unified_pipeline.py's UnifiedDomainPipeline), replacing the
        old separate LLM-only/deterministic-only auto_scan. One payslip
        represents one artifact -> exactly one verdict."""
        verdicts = correct_ingest_response["fraud_verdicts"]
        assert len(verdicts) == 1
        assert verdicts[0]["predicted"] in {"FRAUD", "LEGITIMATE"}
        assert verdicts[0]["kind"] == "payroll"

    def test_correct_payslip_is_not_flagged(self, correct_ingest_response):
        verdicts = correct_ingest_response["fraud_verdicts"]
        assert verdicts[0]["predicted"] == "LEGITIMATE"

    def test_broken_payslip_is_flagged(self, broken_ingest_response):
        """The demo generator silently drops the health deduction from the
        stated net pay -- PayrollDetectorAgent's Tier 2 rules layer
        (GROSS-Deduction==Net_Pay, the same invariant the old
        ValidatorAgent-only check validated) is what has to catch this,
        reached via _map_extraction_to_records() deliberately preferring
        Tier 2's shape over Tier 1's for a payslip (see that method's
        docstring for why Tier 1's rules wouldn't catch this specific,
        realistic error at all)."""
        verdicts = broken_ingest_response["fraud_verdicts"]
        assert verdicts[0]["predicted"] == "FRAUD"
        assert "reconciliation" in verdicts[0]["detector_response"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# CHAT AFTER INGEST  (the point of this whole redesign — no re-upload)
# ══════════════════════════════════════════════════════════════════════════════


class TestChatAfterIngest:
    def test_followup_query_answers_against_same_document(self, pipeline, correct_ingest_response):
        doc = correct_ingest_response["document"]
        result = pipeline.run("What was the gross pay?", source_document=doc)
        assert result["status"] == "valid"
        assert result["data"]["source_document"] == doc
        assert len(result["data"]["answer"].strip()) > 0

    def test_followup_query_draws_on_reference_corpus(self, pipeline, correct_ingest_response):
        """PayrollHrPipeline.REFERENCE_SOURCE is set (see pipeline.py) --
        confirms retrieve() actually merges the ingested payslip with the
        IRS Publication 15-T reference corpus
        (domains/payroll_hr/reference_corpus/irs_pub15t_excerpt.txt,
        run via domains/payroll_hr/data/ingest_reference_corpus.py) rather
        than answering from the payslip alone. Grounded in a real, quoted
        2026 Weekly Wage Bracket Method figure that appears nowhere in the
        payslip itself: a Single filer, no Step 2 checkbox, Adjusted Wage
        Amount $785-$795, has a published Tentative Withholding Amount of
        exactly $53."""
        doc = correct_ingest_response["document"]
        result = pipeline.run(
            "According to the IRS Wage Bracket Method table for a Single "
            "filer paid weekly with the standard withholding column (no "
            "Form W-4 Step 2 checkbox), what is the Tentative Withholding "
            "Amount for an Adjusted Wage Amount between $785 and $795?",
            source_document=doc,
        )
        assert result["status"] == "valid"
        assert "53" in result["data"]["answer"]

    def test_evaluation_present_and_valid(self, pipeline, correct_ingest_response):
        """PayrollHrPipeline.evaluator is now set (domains/payroll_hr/evaluation/
        rag_evaluator.py, a thin wrapper over core/rag/base_rag_evaluator.py).
        Real ground truth for EMP001 lives in that domain's ground_truth.json."""
        doc = correct_ingest_response["document"]
        result = pipeline.run("What was the gross pay?", source_document=doc)
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
