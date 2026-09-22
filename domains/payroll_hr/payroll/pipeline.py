# domains/payroll_hr/payroll/pipeline.py
# PayrollPipeline — the AgentOrchestrator-facing adapter for this domain.
# Same ingest -> [auto capability] / retrieve -> reason -> validate shape as
# domains/banking/documents/pipeline.py; see core/config_loader.py's
# CapabilityConfig and configs/payroll.yaml's `capabilities` section for the
# one-upload-then-chat design this implements.
#
# No RAG-quality evaluator wired here yet (see the plan this was built
# from) — a trivial follow-up copy of
# domains/banking/documents/evaluation/rag_evaluator.py if wanted later.

from __future__ import annotations

import logging

from core.config_loader import DomainConfig
from domains.payroll_hr.payroll.agents.document_loader_agent import DocumentLoaderAgent
from domains.payroll_hr.payroll.agents.retriever_agent import RetrieverAgent
from domains.payroll_hr.payroll.agents.reasoning_agent import ReasoningAgent
from domains.payroll_hr.payroll.agents.validator_agent import ValidatorAgent
from core.rag.seed_history import build_seed_history

logger = logging.getLogger(__name__)


class PayrollPipeline:
    def __init__(self, config: DomainConfig):
        self.config = config
        self.loader = DocumentLoaderAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()

    def ingest(self, pdf_path: str, filename: str | None = None) -> dict:
        """Extracts, chunks, embeds, and stores a payslip/payroll-register
        PDF, then — since this domain declares an "auto" capability
        (payroll_audit) — runs it immediately against what was just
        ingested. The caller gets a compliance verdict back from ingest()
        itself; document_qa (the open chat) is available via run()
        afterward, any number of times, against this same ingestion.

        Also returns `seed_history`: the verdict rendered as the first
        conversation turn, so a caller (the UI) can seed the chat with
        it -- a follow-up like "why was this flagged?" then has the
        actual verdict already in context."""
        result = self.loader.load(pdf_path, filename=filename)
        auto = self.config.auto_capability
        if auto and auto.query:
            scan = self.run(auto.query, source_document=result["document"])
            result["auto_scan"] = {"capability_id": auto.id, "label": auto.label, **scan}
            result["seed_history"] = build_seed_history(auto.query, scan)
        return result

    def run(
        self, query: str, source_document: str | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """Retrieve -> reason -> validate for one query against an
        already-ingested document. `history` (prior turns on this same
        document, oldest first -- see ingest()'s seed_history for the
        shape) lets a follow-up reference earlier answers or the
        auto-scan verdict rather than being answered in isolation."""
        chunks = self.retriever.retrieve(query, source_document=source_document)
        output = self.reasoning.reason(query, chunks, source_document=source_document, history=history)
        validation = self.validator.validate(output)

        if validation["status"] == "invalid":
            return {"status": "invalid", "errors": validation["errors"]}

        return {
            "status": "valid",
            "data": validation["data"].model_dump(),
            "chunks": chunks,
        }
