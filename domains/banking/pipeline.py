# domains/banking/pipeline.py
# BankingPipeline -- unifies domains/banking/fraud (FraudPipeline --
# rules -> tiered ML -> LLM, plus the analyst/router/alert chain, plus
# the Isolation Forest companion signal) and domains/banking/documents
# (DocumentPipeline's RAG agents, RAGAS-evaluated) into ONE pipeline on
# core/unified_pipeline.py's UnifiedDomainPipeline. Any upload (CSV, PDF,
# or DOCX) always gets both a fraud verdict and a chat-ready ingestion.
#
# Neither existing agent's internals change here -- this class is
# composition, not a rewrite: FraudPipeline stays the thing that
# actually calls FraudDetectorAgent/AnalystAgent/RoutingAgent/AlertAgent,
# and the RAG agents stay exactly what the document-chat pipeline already
# runs.
#
# _map_extraction_to_records() -- the one genuinely new piece of logic --
# maps a bank statement's extracted transactions into fraud-detector
# records, ONE PER TRANSACTION (not one per document): "score the
# artifact, not the entity" means each transaction on the statement is
# its own scoreable artifact, exactly as if it had arrived as a CSV row.
# Only "Amount" (and implicitly "hour", which rule_based_filter()
# already defaults to noon when absent) is honestly derivable from a
# real statement -- Tier 1's anonymized V1-V28 PCA features and Tier 2's
# Account_Age/Credit_Score/Previous_Fraud/Device_Used are account-level
# attributes no bank statement states, so detect_tier() correctly and
# HONESTLY falls through to Tier 3 (LLM fallback) for every
# PDF/DOCX-sourced record. Still real value over the old, separate
# LLM-only anomaly_scan: each transaction gets the ACTUAL rules layer
# (catching a genuine $0.00 transaction for free) plus a real per-
# transaction LLM call through the SAME engine the CSV path uses,
# instead of one holistic LLM judgment over the whole statement with no
# rules layer at all.
#
# DOCUMENT_SCORING_CONTEXT (below) exists because of a real measured
# problem, not a hypothetical one: a real test run with no context
# flagged 100% of a genuine statement's transactions as FRAUD, because
# the Tier 3 LLM's only calibration is card-transaction scale ($5-$500
# "typical" -- see FraudPipeline.default_context()). A salary deposit or
# rent payment is routinely thousands of dollars and is NOT inherently
# suspicious on a bank statement the way it would be on a card. This
# uses the exact same `context` parameter FraudDetectorAgent.build_prompt()
# already accepts, populated here for document-sourced records specifically.

from __future__ import annotations

from core.config_loader import DomainConfig
from core.unified_pipeline import UnifiedDomainPipeline
from domains.banking.fraud.pipeline import FraudPipeline
from domains.banking.fraud.agents.detector_agent import FraudDetectorAgent
from domains.banking.documents.agents.document_loader_agent import DocumentLoaderAgent
from domains.banking.documents.agents.retriever_agent import RetrieverAgent
from domains.banking.documents.agents.reasoning_agent import ReasoningAgent
from domains.banking.documents.agents.validator_agent import ValidatorAgent
from domains.banking.documents.evaluation.rag_evaluator import get_evaluator
from core.rag.reference_corpus import REFERENCE_SOURCE_ID

# Purely enumerative on purpose: ReasoningAgent's prompt restricts the
# "transactions" array to those relevant to the question asked, so an
# earlier version that also asked to "flag anything suspicious" got back
# only the suspicious subset (2 of 6 on the test statement) -- the benign
# transactions were then never scored at all. Suspicion is judged
# per-transaction by the fraud engine downstream, not by this query.
EXTRACTION_QUERY = (
    "List EVERY transaction on this statement, all of them and not only "
    "unusual ones, each with its exact date, amount, type, and description. "
    "Do not omit any transaction."
)

# Real, measured finding, not a guess: without this, a genuine test run
# flagged 100% of a real bank statement's transactions as FRAUD -- the
# Tier 3 LLM's only calibration (FraudDetectorAgent.build_prompt() and
# FraudPipeline.default_context()) is card-transaction scale ("typical
# legitimate transactions range from $5 to $500"), which is simply wrong
# for a bank statement's salary deposits, rent, and ATM withdrawals.
DOCUMENT_CONTEXT = (
    "This transaction was extracted from a bank statement, not a card "
    "network. Bank-statement transactions routinely include salary "
    "deposits, rent, and ATM withdrawals in the thousands to tens of "
    "thousands of dollars -- a large amount alone is NOT unusual here the "
    "way it would be for a card transaction. Judge suspicion from pattern "
    "(timing, duplication, round-tripping), not amount size alone."
)


class BankingPipeline(UnifiedDomainPipeline):
    EXTRACTION_QUERY = EXTRACTION_QUERY
    DOCUMENT_SCORING_CONTEXT = DOCUMENT_CONTEXT
    # Opt-in (see core/unified_pipeline.py's REFERENCE_SOURCE docstring) --
    # set here, not inherited as a default, because Banking has actually
    # run domains/banking/data/ingest_reference_corpus.py against its
    # collection (Regulation E unauthorized-transaction/dispute rules --
    # see domains/banking/reference_corpus/regulation_e_excerpt.txt).
    REFERENCE_SOURCE = REFERENCE_SOURCE_ID

    def __init__(self, config: DomainConfig):
        self.config = config
        self.fraud_pipeline = FraudPipeline(config)
        self.loader = DocumentLoaderAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.evaluator = get_evaluator()  # RAG-quality eval on document_qa/statement_review answers -- unique to Banking so far

    def score_record(self, record: dict, context: str | None = None) -> dict:
        """Delegates to the EXISTING FraudPipeline, unchanged -- so a
        transaction reaches the exact same detector -> analyst -> router
        -> alert chain regardless of whether it arrived as a CSV row or
        a PDF/DOCX-extracted transaction."""
        result = self.fraud_pipeline.run(record, context)
        return {
            "predicted": result["predicted"],
            "reason": result["parsed"].get("reason", ""),
            "parsed": result["parsed"],
            "detector_response": result["detector_response"],
            "investigation": result.get("investigation", ""),
            "routing": result.get("routing", ""),
            "alert": result.get("alert", ""),
        }

    def _map_extraction_to_records(self, extraction) -> list[dict]:
        if not extraction.transactions:
            return []
        # "hour" defaults to noon, the same convention rule_based_filter()
        # already uses for a record with no time-of-day info -- a real
        # bank statement doesn't always state one. "Time" (seconds since
        # the FIRST transaction in creditcard.csv's own training set) has
        # no honest equivalent here at all -- it's a dataset-specific
        # feature, not something extractable from any statement -- so
        # this is a neutral placeholder purely to satisfy AnalystAgent's
        # investigation prompt (only reached on a FRAUD verdict), not a
        # value detect_tier()/rules logic gives any weight to.
        return [{"Amount": t.amount, "hour": 12, "Time": 0} for t in extraction.transactions]

    def extract_from_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        """Passthrough -- Banking's existing receipt/cheque image-input
        capability (GPT-4o Vision) lives on FraudPipeline, unchanged."""
        return self.fraud_pipeline.extract_from_image(image_bytes, mime_type)
