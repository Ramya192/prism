# domains/banking/documents/agents/retriever_agent.py
# Subclasses core/rag/base_retriever_agent.py. The four class attributes
# below reproduce this domain's exact original _hyde()/_generate_variants()
# prompt text, character for character -- this domain is LIVE IN
# PRODUCTION and already RAGAS-evaluated, so this refactor is a pure
# structural extraction, not a prompt change. See the base class's
# module docstring.

from core.rag.base_retriever_agent import BaseRetrieverAgent
from domains.banking.documents.settings import Settings


class RetrieverAgent(BaseRetrieverAgent):
    PERSONA = "a bank statement analyst"
    HYDE_DOC_PHRASE = "bank statement"
    HYDE_INSTRUCTION = "Be specific with amounts, dates, transaction types."
    VARIANT_DOC_PHRASE = "a bank statement PDF"

    def __init__(self):
        super().__init__(Settings)
