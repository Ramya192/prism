# domains/financial_services/agents/retriever_agent.py
# Subclasses core/rag/base_retriever_agent.py. Built from scratch this
# pass (like Insurance) -- no reference corpus ingested yet (FinCEN
# virtual-currency guidance and SEC/CFTC market-manipulation guidance are
# candidate sources, not yet researched or fetched).

from core.rag.base_retriever_agent import BaseRetrieverAgent
from domains.financial_services.settings import Settings


class RetrieverAgent(BaseRetrieverAgent):
    PERSONA = "a fintech/digital-asset fraud analyst"
    HYDE_DOC_PHRASE = "wallet or exchange activity statement"
    HYDE_INSTRUCTION = (
        "Be specific with transaction amounts, dates, transaction types "
        "(send/receive/swap/bridge), blockchains, and counterparties."
    )
    VARIANT_DOC_PHRASE = "a crypto wallet or exchange activity statement"

    def __init__(self):
        super().__init__(Settings)
