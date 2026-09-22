# domains/banking/documents/pipeline.py
# DocumentPipeline — the AgentOrchestrator-facing adapter for this domain.
# Thin wrapper around the exact retrieve -> reason -> validate -> evaluate
# sequence that document_intelligence_system/app.py's and main.py's
# /analyze flow already used, plus an ingest() method for the loader step.
# Returns structured dicts instead of rendering/streaming SSE, so
# streamlit_app.py, main.py, and tests can all share one code path.

from __future__ import annotations

import logging

from core.config_loader import DomainConfig
from domains.banking.documents.agents.document_loader_agent import DocumentLoaderAgent
from domains.banking.documents.agents.retriever_agent import RetrieverAgent
from domains.banking.documents.agents.reasoning_agent import ReasoningAgent
from domains.banking.documents.agents.validator_agent import ValidatorAgent
from domains.banking.documents.evaluation.rag_evaluator import get_evaluator
from core.rag.seed_history import build_seed_history

logger = logging.getLogger(__name__)


class DocumentPipeline:
    def __init__(self, config: DomainConfig):
        self.config = config
        self.loader = DocumentLoaderAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.evaluator = get_evaluator()

    def ingest(self, pdf_path: str, filename: str | None = None) -> dict:
        """Extracts, chunks, embeds, and stores a bank statement PDF, then —
        if this domain declares an "auto" capability (anomaly_scan) — runs
        it immediately against what was just ingested. This is the one-
        upload fix: the caller gets a verdict back from ingest() itself,
        with no need to ask a question first or re-upload to get one.
        document_qa (the open chat) stays available via run() afterward,
        any number of times, against this same ingested document.

        Also returns `seed_history` when the auto-scan ran: the verdict
        rendered as the first conversation turn, so a caller (the UI) can
        seed the chat with it. A follow-up like "why was this flagged?"
        then has the actual verdict already in context instead of the
        model re-deriving one from scratch via a fresh, disconnected
        retrieval."""
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
        """Retrieve -> reason -> validate -> evaluate for one query against
        an already-ingested document. Mirrors main.py's /analyze endpoint
        and streamlit_app.py's analysis flow. `history` (prior turns on
        this same document, oldest first -- see ingest()'s seed_history
        for the shape) lets a follow-up question reference earlier answers
        or the auto-scan verdict rather than being answered in isolation."""
        chunks = self.retriever.retrieve(query, source_document=source_document)
        output = self.reasoning.reason(query, chunks, source_document=source_document, history=history)
        validation = self.validator.validate(output)

        if validation["status"] == "invalid":
            return {"status": "invalid", "errors": validation["errors"]}

        evaluation = None
        try:
            eval_result = self.evaluator.evaluate(
                query=query,
                chunks=chunks,
                answer=output.answer,
                source_document=source_document,
            )
            evaluation = eval_result.to_dict()
        except Exception as e:
            logger.warning(f"RAG evaluation skipped: {e}")

        return {
            "status": "valid",
            "data": validation["data"].model_dump(),
            "evaluation": evaluation,
            "chunks": chunks,
        }
