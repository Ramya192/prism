# domains/payroll_hr/pipeline.py
# PayrollHrFraudPipeline — the AgentOrchestrator-facing adapter for the
# merged Payroll & HR domain's fraud-detection capability (domain id
# "payroll_hr" in configs/payroll_hr.yaml). Payroll-register fraud
# (PayrollDetectorAgent, domains/payroll_hr/payroll/) and job-posting
# fraud (HrDetectorAgent, domains/payroll_hr/hr/) were built as two
# separate detectors on two genuinely different record shapes; this
# class is what actually makes them "one domain" rather than two
# configs glued together by a shared directory.
#
# Routing is CONTENT-based, never by upload format or by a hint the
# caller passes in -- see docs/UNIFIED_INGESTION_VISION.md: a record's
# own fields decide which detector applies, the same way
# DomainClassifierAgent decides which domain applies. A payroll-register
# row and a job posting share zero field names, so this is a clean,
# unambiguous split in practice -- see _select_detector()'s docstring
# for the (rare) undecidable case.
#
# PayrollHrPipeline (below) folds this domain's RAG/document side
# (payslip PDF/DOCX chat) into the SAME pipeline, on
# core/unified_pipeline.py's UnifiedDomainPipeline -- the same treatment
# Banking (domains/banking/pipeline.py) and Insurance use. This class
# (PayrollHrFraudPipeline) is reused as a component here, not rewritten
# -- the same pattern Banking uses with FraudPipeline.

from __future__ import annotations

from core.config_loader import DomainConfig
from core.unified_pipeline import UnifiedDomainPipeline
from core.rag.reference_corpus import REFERENCE_SOURCE_ID
from domains.payroll_hr.evaluation.rag_evaluator import get_evaluator
from domains.payroll_hr.payroll.agents.payroll_detector_agent import PayrollDetectorAgent
from domains.payroll_hr.hr.agents.hr_detector_agent import HrDetectorAgent
from domains.payroll_hr.payroll.agents.document_loader_agent import DocumentLoaderAgent
from domains.payroll_hr.payroll.agents.retriever_agent import RetrieverAgent
from domains.payroll_hr.payroll.agents.reasoning_agent import ReasoningAgent
from domains.payroll_hr.payroll.agents.validator_agent import ValidatorAgent
from domains.payroll_hr.payroll.agents.validator_agent import _DEDUCTION_CATEGORIES

DEFAULT_PAYROLL_CSV = "domains/payroll_hr/payroll/data/tier_ml_train.csv"
DEFAULT_HR_CSV = "domains/payroll_hr/hr/data/train.csv"

# Fields that only ever appear on one side or the other -- used for the
# weak-signal tie-break in _select_detector() when neither detector's own
# has_recognizable_schema()/detect_tier() fully matches (both would fall
# through to their own tier3/LLM path anyway in that case).
_PAYROLL_SIGNAL_FIELDS = {
    "BasePay", "OvertimePay", "OtherPay", "Stated_TotalPay", "Stated_TotalPayBenefits",
    "GROSS", "Deduction", "Net_Pay", "Benefits",
}
_POSTING_SIGNAL_FIELDS = {
    "title", "description", "employment_type", "required_experience",
    "required_education", "telecommuting", "has_company_logo", "has_questions",
    "industry", "function", "salary_range",
}


class PayrollHrFraudPipeline:
    def __init__(self, config: DomainConfig):
        self.config = config
        data = config.extra.get("data", {})
        payroll_csv = data.get("payroll_demo_csv", DEFAULT_PAYROLL_CSV)
        hr_csv = data.get("hr_demo_csv", DEFAULT_HR_CSV)

        self.payroll_detector = PayrollDetectorAgent(name="PayrollDetectorAgent-1", data_path=payroll_csv)
        self.hr_detector = HrDetectorAgent(name="HrDetectorAgent-1", data_path=hr_csv)

    @staticmethod
    def _looks_like_payroll(record: dict) -> bool:
        return any(f in record for f in _PAYROLL_SIGNAL_FIELDS)

    @staticmethod
    def _looks_like_job_posting(record: dict) -> bool:
        return any(f in record for f in _POSTING_SIGNAL_FIELDS)

    def _select_detector(self, record: dict):
        """Content decides, never file type or a caller-supplied hint.
        Prefers a FULL schema match (detect_tier()/has_recognizable_schema())
        over the weak any-field signal below, since a full match means the
        record's rules+ML layers actually apply -- not just its LLM
        fallback. Only falls back to the weak signal when neither detector
        recognizes the schema at all (both would go straight to their own
        tier3/LLM path either way, so which one merely picks the framing
        of that LLM call). Defaults to payroll on a genuinely undecidable
        record (no signal fields from either side) -- an edge case
        DomainClassifierAgent routing into "payroll_hr" in the first place
        should make rare in practice, not a meaningful design choice."""
        if PayrollDetectorAgent.has_recognizable_schema(record):
            return self.payroll_detector, "payroll"
        if HrDetectorAgent.has_recognizable_schema(record):
            return self.hr_detector, "hr"
        if self._looks_like_job_posting(record) and not self._looks_like_payroll(record):
            return self.hr_detector, "hr"
        return self.payroll_detector, "payroll"

    def run(self, record: dict, context: str | None = None) -> dict:
        """Routes one record to whichever detector its own fields match,
        then runs that detector's full rules -> ML -> LLM flow. Both
        detectors share the same Risk Level/Reason/Action response
        contract (parse_response/is_fraud), so the caller doesn't need to
        know which one actually ran except via the returned "kind"."""
        detector, kind = self._select_detector(record)
        response = detector.analyse(record, context)
        agent_cls = PayrollDetectorAgent if kind == "payroll" else HrDetectorAgent
        parsed = agent_cls.parse_response(response)
        predicted = "FRAUD" if agent_cls.is_fraud(parsed) else "LEGITIMATE"
        return {
            "kind": kind,
            "predicted": predicted,
            "parsed": parsed,
            "detector_response": response,
        }


EXTRACTION_QUERY = (
    "List every earning and deduction line item on this payslip with its "
    "exact amount, plus the gross pay and net pay totals, and flag "
    "anything that looks miscalculated — an overtime rate that isn't 1.5x "
    "the regular rate, a missing required withholding line, a duplicate "
    "line item, or deductions that don't add up to the difference between "
    "gross and net pay."
)


HANDBOOK_QUERY = "What is this document and what are its key sections or policies?"


class PayrollHrPipeline(UnifiedDomainPipeline):
    """The domain's actual pipeline (configs/payroll_hr.yaml points here,
    not at PayrollHrFraudPipeline directly) -- composes the untouched
    PayrollHrFraudPipeline (CSV/manual-entry, content-routed between
    payroll and HR) with Payroll's RAG agents (payslip PDF/DOCX chat).

    PDF/DOCX uploads here are NOT unambiguously payslips. Relying only on
    _map_extraction_to_records() returning [] for a non-payslip document
    (the base class's documented graceful-degradation behavior) isn't
    enough on its own: a real e2e test against an actual HR employee
    handbook (tests/test_payroll_hr_handbook_pipeline.py) shows that when
    asked EXTRACTION_QUERY's payslip-specific question, the LLM doesn't
    reliably return nulls -- it can fabricate a plausible-looking
    GROSS/Deduction/Net_Pay record with dollar figures that appear
    NOWHERE in the source document, which _map_extraction_to_records()
    would then honestly (and wrongly) turn into a real FRAUD verdict on a
    document that was never a payslip to begin with.

    The actual guard is core/unified_pipeline.py's document-type gate
    (DOCUMENT_TYPE_DESCRIPTION below): a cheap content classification
    BEFORE the payslip extraction query ever runs, so a non-payslip
    document never reaches a prompt that pressures the LLM into inventing
    payslip-shaped numbers. This domain's handbook bug is where the gate
    came from; every domain now shares it. It is the document-upload
    version of the same "content decides, not file type or a hint"
    principle _select_detector() above already applies to CSV/manual-
    entry records."""

    EXTRACTION_QUERY = EXTRACTION_QUERY
    DOCUMENT_TYPE_DESCRIPTION = (
        "an individual employee's PAYSLIP or PAY STUB -- a specific pay period's stated "
        "gross pay, deductions, and net pay for one named employee"
    )
    NON_MATCH_EXAMPLES = "a general HR policy document, employee handbook, benefits guide, or job posting"
    NON_MATCH_QUERY = HANDBOOK_QUERY
    # Opt-in (see core/unified_pipeline.py's REFERENCE_SOURCE docstring) --
    # set here, not inherited as a default, because Payroll & HR has
    # actually run domains/payroll_hr/data/ingest_reference_corpus.py
    # against its collection (IRS Publication 15-T withholding tables --
    # see domains/payroll_hr/reference_corpus/irs_pub15t_excerpt.txt).
    REFERENCE_SOURCE = REFERENCE_SOURCE_ID

    def __init__(self, config: DomainConfig):
        self.config = config
        self.fraud_pipeline = PayrollHrFraudPipeline(config)
        self.loader = DocumentLoaderAgent()
        self.retriever = RetrieverAgent()
        self.reasoning = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.evaluator = get_evaluator()  # RAG-quality eval on document_qa/payroll_audit answers -- core/rag/base_rag_evaluator.py

    def score_record(self, record: dict, context: str | None = None) -> dict:
        """Delegates to the EXISTING PayrollHrFraudPipeline, unchanged --
        a payslip-derived record still goes through its own content-based
        routing (which will always land on the payroll detector for a
        payroll-shaped record, but reuses that tested logic rather than
        calling PayrollDetectorAgent directly)."""
        result = self.fraud_pipeline.run(record, context)
        return {
            "predicted": result["predicted"],
            "reason": result["parsed"].get("reason", ""),
            "kind": result["kind"],
            "parsed": result["parsed"],
            "detector_response": result["detector_response"],
        }

    def _map_extraction_to_records(self, extraction) -> list[dict]:
        """A payslip represents one artifact -> at most one record (same
        as Insurance's claim mapping, unlike Banking's many-transactions-
        per-statement case).

        Tries Tier 2's GROSS/Deduction/Net_Pay shape FIRST, not Tier 1 --
        deliberately, not just "whichever schema matches": a payslip's
        actual compliance question is "do the deductions reconcile gross
        to net", exactly what Tier 2's rules layer checks
        (GROSS-Deduction==Net_Pay), and exactly the invariant this
        project's own demo generator (generate_payroll_register.py)
        breaks for its "broken" test payslip -- the health deduction is
        computed and listed, but silently dropped from the stated net
        pay, while gross pay's OWN composition (base+overtime) stays
        internally consistent throughout. Tier 1's BasePay/OvertimePay/
        OtherPay/Stated_TotalPay rules check a DIFFERENT invariant (does
        gross pay's makeup add up), which wouldn't catch this specific,
        realistic error at all. gross_pay/net_pay are always top-level
        extraction fields (not dependent on which line-item categories
        appear), so this is also the more reliably derivable mapping.
        Falls back to attempting Tier 1's shape only if gross_pay/net_pay
        are somehow missing but a full Regular-Hours/Overtime breakdown
        is present. Returns [] if neither shape is honestly derivable."""
        line_items = {li.category: li.amount for li in extraction.line_items}

        if extraction.gross_pay is not None and extraction.net_pay is not None:
            deduction_total = sum(amt for cat, amt in line_items.items() if cat in _DEDUCTION_CATEGORIES)
            return [{
                "GROSS": extraction.gross_pay,
                "Deduction": deduction_total,
                "Net_Pay": extraction.net_pay,
            }]

        if "Regular Hours" in line_items and "Overtime" in line_items:
            return [{
                "BasePay": line_items["Regular Hours"],
                "OvertimePay": line_items["Overtime"],
                "OtherPay": 0.0,  # no "other pay" category exists in this extraction schema at all
                "Stated_TotalPay": line_items["Regular Hours"] + line_items["Overtime"],
            }]

        return []
