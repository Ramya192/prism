# domains/financial_services/pipeline.py
# FinancialServicesPipeline -- combines fraud detection
# (FintechDetectorAgent -- rules -> tiered ML -> LLM) with a
# document/RAG capability on core/unified_pipeline.py's
# UnifiedDomainPipeline. Any upload (CSV, PDF, or DOCX) always gets both
# a fraud verdict and a chat-ready ingestion.
#
# Honest limit, stated plainly rather than glossed over: NEITHER fraud
# tier's REQUIRED_COLS can ever be satisfied by a real document.
# Tier 1 needs sender_wallet_age_days (an account-level attribute, not a
# per-transaction line item); Tier 2 needs round_trip_score and
# counterparty_reuse_ratio (backend risk-engine outputs no statement
# would ever print). So _map_extraction_to_records() always resolves to
# Tier 3 -- same honest degradation as Banking's PDF path, but with one
# real difference worth stating plainly: FintechDetectorAgent's rules
# layer is TIER-GATED (`if tier != "tier3":` wraps both rules and ML
# together -- see analyse() below), unlike Banking's FraudDetectorAgent,
# which deliberately decouples rules from ML-tier eligibility. That means
# the malformed-record rule check NEVER fires on a document-extracted
# transaction here, only on CSV rows. This domain's document path's real
# value is narrower than Banking's as a result: real LLM reasoning
# through the actual engine and persona, with every extracted field
# (transaction type, blockchain, counterparty, cross-chain flag)
# enriching that reasoning -- not a free rules-layer catch.
#
# DOCUMENT_SCORING_CONTEXT is set proactively -- the same fix Banking's
# document path needed: an LLM tier with no context, calibrated only on
# card-transaction scale, will flag effectively 100% of a statement's
# transactions since nothing about the calibration matches statement-scale
# amounts.
# FintechDetectorAgent's own tier3 prompt has no such hardcoded bias, but
# there's no reason to leave this domain's LLM without a scale hint
# either, given the same class of failure is now a known, documented risk.

from __future__ import annotations

from core.config_loader import DomainConfig
from core.unified_pipeline import UnifiedDomainPipeline
from domains.financial_services.agents.fintech_detector_agent import FintechDetectorAgent
from domains.financial_services.agents.document_loader_agent import DocumentLoaderAgent
from domains.financial_services.agents.retriever_agent import RetrieverAgent
from domains.financial_services.agents.reasoning_agent import ReasoningAgent
from domains.financial_services.agents.validator_agent import ValidatorAgent
from core.rag.reference_corpus import REFERENCE_SOURCE_ID
from domains.financial_services.evaluation.rag_evaluator import get_evaluator

DEFAULT_DEMO_CSV = "domains/financial_services/data/train.csv"
DEFAULT_TIER2_CSV = "domains/financial_services/data/tier2_train.csv"

EXTRACTION_QUERY = (
    "List every transaction on this statement with its exact date, amount, "
    "transaction type, blockchain, and counterparty, and flag anything "
    "suspicious — duplicate transactions, a counterparty appearing twice "
    "with conflicting amounts, or an internally inconsistent total."
)

DOCUMENT_CONTEXT = (
    "This transaction was extracted from a wallet or exchange activity "
    "statement, not a card network. Crypto transactions routinely range "
    "from a few dollars to tens of thousands of dollars depending on the "
    "token and purpose -- a large amount alone is NOT unusual here. Judge "
    "suspicion from pattern (duplication, rapid succession, unfamiliar "
    "counterparties), not amount size alone."
)


class FinancialServicesPipeline(UnifiedDomainPipeline):
    EXTRACTION_QUERY = EXTRACTION_QUERY
    DOCUMENT_SCORING_CONTEXT = DOCUMENT_CONTEXT
    DOCUMENT_TYPE_DESCRIPTION = (
        "a crypto wallet or exchange activity statement -- a list of individual transfers, "
        "deposits, withdrawals, or trades"
    )
    NON_MATCH_EXAMPLES = "a whitepaper, a terms-of-service or regulatory guidance document, or a marketing page"
    # Opt-in (see core/unified_pipeline.py's REFERENCE_SOURCE docstring) --
    # set here, not inherited as a default, because Financial Services has
    # actually run domains/financial_services/data/ingest_reference_corpus.py
    # against its collection (FinCEN CVC guidance FIN-2019-G001 -- see
    # domains/financial_services/reference_corpus/fincen_cvc_guidance_excerpt.txt).
    REFERENCE_SOURCE = REFERENCE_SOURCE_ID

    def __init__(self, config: DomainConfig):
        self.config = config
        data = config.extra.get("data", {})
        data_path = data.get("demo_csv", DEFAULT_DEMO_CSV)
        tier2_data_path = data.get("tier2_demo_csv", DEFAULT_TIER2_CSV)
        self.detector = FintechDetectorAgent(
            name="FintechDetectorAgent-1", data_path=data_path, tier2_data_path=tier2_data_path,
        )
        self.loader = DocumentLoaderAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.evaluator = get_evaluator()  # RAG-quality eval on document_qa/statement_review answers -- core/rag/base_rag_evaluator.py

    def score_record(self, record: dict, context: str | None = None) -> dict:
        """Delegates to the EXISTING FintechDetectorAgent, unchanged --
        same rules -> tiered ML -> LLM flow regardless of whether the
        record arrived as a CSV row or a statement-extracted transaction."""
        response = self.detector.analyse(record, context)
        parsed = FintechDetectorAgent.parse_response(response)
        predicted = "FRAUD" if FintechDetectorAgent.is_fraud(parsed) else "LEGITIMATE"
        return {
            "predicted": predicted,
            "reason": parsed.get("reason", ""),
            "parsed": parsed,
            "detector_response": response,
        }

    def _map_extraction_to_records(self, extraction) -> list[dict]:
        """One record per transaction (a statement lists many, like
        Banking's -- not one artifact like Insurance's claim or Payroll's
        payslip). Only "transaction_amount_usd" is a detect_tier()-
        recognized field name -- see the module docstring for why neither
        ML tier's other required fields can ever be honestly derived from
        a real document, so this always resolves to Tier 3 regardless of
        what else is in the dict. The other extracted fields ARE still
        included here anyway (not just described in a context string):
        FintechDetectorAgent.build_prompt()'s tier3 branch renders every
        key a record actually has into the LLM prompt, so a genuinely
        richer record produces genuinely better Tier 3 reasoning, even
        though none of these extra keys can ever change the tier decision
        itself."""
        return [
            {
                "transaction_amount_usd": t.amount_usd,
                "transaction_type": t.transaction_type,
                "blockchain": t.blockchain,
                "counterparty": t.counterparty,
                "is_cross_chain": t.is_cross_chain,
            }
            for t in extraction.transactions
        ]
