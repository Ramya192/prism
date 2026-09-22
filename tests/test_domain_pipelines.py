"""
tests/test_domain_pipelines.py
Tests for InsurancePipeline (domain id "insurance") and PayrollHrPipeline
(domain id "payroll_hr") -- both built on core/unified_pipeline.py's
UnifiedDomainPipeline, composing their EXISTING, untouched fraud
detectors (HealthcareDetectorAgent / PayrollHrFraudPipeline's
payroll+HR content-router) via score_record(), distinct from the RAG
chat run(query, ...) both classes now also expose. Verifies the config
-> orchestrator -> pipeline -> detector chain actually works end to end,
not just the detector agents in isolation (already covered by
test_healthcare_detector.py / test_payroll_detector.py / test_hr_compliance_detector.py).

A dummy OPENAI_API_KEY is enough -- these cases all resolve at the rules
layer, no LLM call needed.
"""

from core.orchestrator import AgentOrchestrator


class TestInsurancePipeline:
    def test_reachable_via_orchestrator(self):
        orchestrator = AgentOrchestrator()
        pipeline = orchestrator.get_pipeline("insurance")
        assert pipeline is not None

    def test_score_record_resolves_rule_violation(self):
        # score_record() -- Insurance's direct single-record fraud-scoring
        # entry point, distinct from the RAG chat run(query, ...) both now
        # share one InsurancePipeline (UnifiedDomainPipeline) class.
        orchestrator = AgentOrchestrator()
        pipeline = orchestrator.get_pipeline("insurance")
        claim = {
            "Claim_Amount": 500.0, "Approved_Amount": 9999.0,
            "Days_Between_Service_and_Claim": 15,
        }
        result = pipeline.score_record(claim)
        assert result["predicted"] == "FRAUD"
        assert "exceeds the claimed amount" in result["detector_response"]


class TestPayrollHrPipeline:
    def test_reachable_via_orchestrator(self):
        orchestrator = AgentOrchestrator()
        pipeline = orchestrator.get_pipeline("payroll_hr")
        assert pipeline is not None

    def test_job_posting_record_routes_to_hr_detector(self):
        orchestrator = AgentOrchestrator()
        pipeline = orchestrator.get_pipeline("payroll_hr")
        posting = {
            "title": "", "description": "x",
            "telecommuting": 0, "has_company_logo": 1, "has_questions": 0,
        }
        result = pipeline.score_record(posting)
        assert result["kind"] == "hr"
        assert result["predicted"] == "FRAUD"
        assert "no title" in result["detector_response"]

    def test_payroll_register_record_routes_to_payroll_detector(self):
        # Tier 1 schema (BasePay/OvertimePay/OtherPay/Stated_TotalPay) with
        # a deliberately broken reconciliation -- resolves at the rules
        # layer, same as the HR case above.
        orchestrator = AgentOrchestrator()
        pipeline = orchestrator.get_pipeline("payroll_hr")
        record = {
            "BasePay": 5000.0, "OvertimePay": 200.0, "OtherPay": 0.0,
            "Stated_TotalPay": 9999.0,  # doesn't match 5000+200+0
        }
        result = pipeline.score_record(record)
        assert result["kind"] == "payroll"
        assert result["predicted"] == "FRAUD"
        assert "Reconciliation error" in result["detector_response"]
