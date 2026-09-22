# domains/banking/documents/agents/document_loader_agent.py
# Subclasses core/rag/base_document_loader_agent.py -- the only thing
# specific to this domain is bank-name detection via FormatDetector,
# now a single _extract_metadata() override instead of duplicated
# chunk/embed/store logic. See the base class's module docstring: this
# domain is LIVE IN PRODUCTION and already RAGAS-evaluated, so this
# refactor changes zero storage/chunking behavior -- same payload shape
# (source, text, chunk_index, bank_name), same everything, just the
# mechanics moved to a shared base once payroll's near-identical copy
# made the duplication undeniable.

from core.rag.base_document_loader_agent import BaseDocumentLoaderAgent
from domains.banking.documents.settings import Settings
from domains.banking.documents.tools.format_detector import FormatDetector


class DocumentLoaderAgent(BaseDocumentLoaderAgent):
    def __init__(self):
        super().__init__(Settings)

    def _extract_metadata(self, full_text: str) -> dict:
        return {"bank_name": FormatDetector().detect(full_text)}
