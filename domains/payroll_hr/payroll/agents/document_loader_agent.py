# domains/payroll_hr/payroll/agents/document_loader_agent.py
# Subclasses core/rag/base_document_loader_agent.py -- no metadata
# override needed (a payroll register doesn't need a bank-name detector,
# so the base's default empty _extract_metadata() is used as-is).

from core.rag.base_document_loader_agent import BaseDocumentLoaderAgent
from domains.payroll_hr.payroll.settings import Settings


class DocumentLoaderAgent(BaseDocumentLoaderAgent):
    def __init__(self):
        super().__init__(Settings)
