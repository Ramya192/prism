# domains/insurance/pipeline.py
# InsurancePipeline -- built on UnifiedDomainPipeline. One config id
# ("insurance"), one pipeline: any upload (CSV/PDF/DOCX) always gets both
# a fraud verdict and a chat-ready ingestion (see
# docs/UNIFIED_INGESTION_VISION.md).
#
# self.detector is HealthcareDetectorAgent (rules -> tiered ML -> LLM,
# AUC 0.998/0.857 on Tier1/Tier2 respectively). A PDF/DOCX claim's
# LLM-extracted fields route into that same detector via
# _map_extraction_to_records(), so a document-sourced claim gets the
# identical fraud-scoring a CSV row would.
#
# _map_extraction_to_records() returns [] (no forced/fake match) unless
# the extraction actually produced Claim_Amount, Approved_Amount, AND
# both dates needed to compute Days_Between_Service_and_Claim --
# detect_tier() then correctly falls through to Tier 3 (LLM fallback) on
# a claim document that's missing one of those. A claim/EOB document
# represents exactly one artifact, so this returns at most one record
# (unlike Banking, where one statement lists many transactions) -- see
# core/unified_pipeline.py's module docstring for why that's a per-domain
# decision, not a fixed one-per-document rule.

from __future__ import annotations

from datetime import date

from core.config_loader import DomainConfig
from core.unified_pipeline import UnifiedDomainPipeline
from core.rag.reference_corpus import REFERENCE_SOURCE_ID
from domains.insurance.agents.healthcare_detector_agent import HealthcareDetectorAgent
from domains.insurance.agents.document_loader_agent import DocumentLoaderAgent
from domains.insurance.agents.retriever_agent import RetrieverAgent
from domains.insurance.agents.reasoning_agent import ReasoningAgent
from domains.insurance.agents.validator_agent import ValidatorAgent
from domains.insurance.evaluation.rag_evaluator import get_evaluator

DEFAULT_DEMO_CSV = "domains/insurance/data/train.csv"

EXTRACTION_QUERY = (
    "List the claim amount, approved amount, service date, claim date, "
    "patient age and gender, insurance type, provider specialty, and "
    "visit type, and flag anything that looks inconsistent — an approved "
    "amount larger than the billed amount, or a claim date before the "
    "service date."
)


class InsurancePipeline(UnifiedDomainPipeline):
    EXTRACTION_QUERY = EXTRACTION_QUERY
    # Opt-in (see core/unified_pipeline.py's REFERENCE_SOURCE docstring) --
    # set here, not inherited as a default, because Insurance has actually
    # run domains/insurance/data/ingest_reference_corpus.py against its
    # collection. Domains without one (Banking, so far) must NOT set this
    # until they've done the same, or retrieve() will filter on a source
    # that was never ingested.
    REFERENCE_SOURCE = REFERENCE_SOURCE_ID

    def __init__(self, config: DomainConfig):
        self.config = config
        data_path = config.extra.get("data", {}).get("demo_csv", DEFAULT_DEMO_CSV)
        self.detector = HealthcareDetectorAgent(name="HealthcareDetectorAgent-1", data_path=data_path)
        self.loader = DocumentLoaderAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.evaluator = get_evaluator()  # RAG-quality eval on document_qa/claim_review answers -- core/rag/base_rag_evaluator.py

    def score_record(self, record: dict, context: str | None = None) -> dict:
        response = self.detector.analyse(record, context)
        parsed = HealthcareDetectorAgent.parse_response(response)
        predicted = "FRAUD" if HealthcareDetectorAgent.is_fraud(parsed) else "LEGITIMATE"
        return {
            "predicted": predicted,
            "reason": parsed.get("reason", ""),
            "parsed": parsed,
            "detector_response": response,
        }

    def _map_extraction_to_records(self, extraction) -> list[dict]:
        if extraction.Claim_Amount is None or extraction.Approved_Amount is None:
            return []

        record = {
            "Claim_Amount": extraction.Claim_Amount,
            "Approved_Amount": extraction.Approved_Amount,
        }

        if extraction.Service_Date and extraction.Claim_Date:
            try:
                service = date.fromisoformat(extraction.Service_Date)
                claimed = date.fromisoformat(extraction.Claim_Date)
                record["Days_Between_Service_and_Claim"] = (claimed - service).days
            except ValueError:
                pass  # dates present but unparseable -- leave the field out, don't guess

        if extraction.Patient_Age is not None:
            record["Patient_Age"] = extraction.Patient_Age
        if extraction.Patient_Gender:
            record["Patient_Gender"] = extraction.Patient_Gender
        if extraction.Insurance_Type:
            record["Insurance_Type"] = extraction.Insurance_Type
        if extraction.Provider_Specialty:
            record["Provider_Specialty"] = extraction.Provider_Specialty
        if extraction.Visit_Type:
            record["Visit_Type"] = extraction.Visit_Type

        return [record]
