# domains/financial_services/agents/document_loader_agent.py
# Subclasses core/rag/base_document_loader_agent.py -- no metadata
# override needed. Accepts PDF/DOCX/CSV like every other document-capable
# domain now does -- see the base class for the format dispatch.

from core.rag.base_document_loader_agent import BaseDocumentLoaderAgent
from domains.financial_services.settings import Settings


class DocumentLoaderAgent(BaseDocumentLoaderAgent):
    def __init__(self):
        super().__init__(Settings)
