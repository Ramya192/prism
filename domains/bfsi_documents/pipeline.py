# domains/bfsi_documents/pipeline.py
# DocumentPipeline — the AgentOrchestrator-facing adapter for this domain.
# Thin wrapper around the exact retrieve -> reason -> validate -> evaluate
# sequence that document_intelligence_system/app.py's and main.py's
# /analyze flow already used, plus an ingest() method for the loader step.
# Returns structured dicts instead of rendering/streaming SSE, so
# streamlit_app.py, main.py, and tests can all share one code path.

from __future__ import annotations

import logging

from core.config_loader import DomainConfig
from domains.bfsi_documents.agents.document_loader_agent import DocumentLoaderAgent
from domains.bfsi_documents.agents.retriever_agent import RetrieverAgent
from domains.bfsi_documents.agents.reasoning_agent import ReasoningAgent
from domains.bfsi_documents.agents.validator_agent import ValidatorAgent
from domains.bfsi_documents.evaluation.rag_evaluator import get_evaluator

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
        """Extracts, chunks, embeds, and stores a bank statement PDF."""
        return self.loader.load(pdf_path, filename=filename)

    def run(self, query: str, source_document: str | None = None) -> dict:
        """Retrieve -> reason -> validate -> evaluate for one query against
        an already-ingested document. Mirrors main.py's /analyze endpoint
        and streamlit_app.py's analysis flow."""
        chunks = self.retriever.retrieve(query, source_document=source_document)
        output = self.reasoning.reason(query, chunks, source_document=source_document)
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
