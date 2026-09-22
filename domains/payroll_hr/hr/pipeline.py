# domains/payroll_hr/hr/pipeline.py
# HrCompliancePipeline — the AgentOrchestrator-facing adapter for this
# domain's fraud-detection capability (job-posting screening), same
# thin-wrapper shape as bfsi_fraud's FraudPipeline. Wraps
# HrDetectorAgent's rules -> ML -> LLM posting analysis.
#
# This domain's second, planned capability (policy-document RAG over
# employee handbooks) has no pipeline yet -- when it's built, this file
# is the place a document_qa-style capability would be added, the same
# way payroll's PayrollPipeline (RAG) and PayrollDetectorAgent (CSV
# fraud) are two separate things sharing one domain id.

from __future__ import annotations

from core.config_loader import DomainConfig
from domains.payroll_hr.hr.agents.hr_detector_agent import HrDetectorAgent

DEFAULT_DEMO_CSV = "domains/payroll_hr/hr/data/train.csv"


class HrCompliancePipeline:
    def __init__(self, config: DomainConfig):
        self.config = config
        data_path = config.extra.get("data", {}).get("demo_csv", DEFAULT_DEMO_CSV)
        self.detector = HrDetectorAgent(name="HrDetectorAgent-1", data_path=data_path)

    def run(self, posting: dict, context: str | None = None) -> dict:
        """Runs rules -> ML -> LLM for one job posting. See
        HrDetectorAgent.analyse() for the full decision flow."""
        response = self.detector.analyse(posting, context)
        parsed = HrDetectorAgent.parse_response(response)
        predicted = "FRAUD" if HrDetectorAgent.is_fraud(parsed) else "LEGITIMATE"
        return {
            "predicted": predicted,
            "parsed": parsed,
            "detector_response": response,
        }
