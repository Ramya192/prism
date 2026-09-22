# domains/insurance/agents/retriever_agent.py
# Subclasses core/rag/base_retriever_agent.py. First domain to be built
# from scratch on the unified pipeline -- retrieval draws on this
# document AND the domain's reference corpus (CMS Claims Processing
# Manual excerpt) in the same fused ranking by default, via
# UnifiedDomainPipeline.run()'s reference_source param.

from core.rag.base_retriever_agent import BaseRetrieverAgent
from domains.insurance.settings import Settings


class RetrieverAgent(BaseRetrieverAgent):
    PERSONA = "an insurance claims compliance analyst"
    HYDE_DOC_PHRASE = "insurance claim / EOB"
    HYDE_INSTRUCTION = (
        "Be specific with claim amounts, approved amounts, dates of service, "
        "diagnosis/procedure codes, and provider or patient details."
    )
    VARIANT_DOC_PHRASE = "an insurance claim document or Explanation of Benefits (EOB)"

    def __init__(self):
        super().__init__(Settings)
