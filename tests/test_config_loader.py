# tests/test_config_loader.py
# Verifies ConfigLoader/AgentOrchestrator/DomainClassifierAgent work
# against the real configs/*.yaml, and that stub domains are correctly
# excluded from anything that tries to run them.
#
# Domain ids, each a single unified pipeline (fraud detection AND
# document chat together, never split across two ids): banking
# (bank-statement document chat), payroll_hr (payslip document chat +
# job-posting fraud detection), insurance (claim/EOB document chat),
# financial_services (wallet/exchange statement document chat -- see
# domains/financial_services/pipeline.py's module docstring for why a
# document's extracted transactions always resolve to the LLM-fallback
# tier).

import pytest

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator
from core.domain_classifier_agent import DomainClassifierAgent

ALL_DOMAIN_IDS = {"banking", "payroll_hr", "insurance", "financial_services"}


@pytest.fixture(scope="module")
def loader():
    return ConfigLoader()


class TestConfigLoader:
    def test_loads_all_four_configs(self, loader):
        ids = {d.id for d in loader.list_domains()}
        assert ids == ALL_DOMAIN_IDS

    def test_working_domains_are_runnable(self, loader):
        # All 4 business domains are now each a single unified pipeline
        # (or, for financial_services, fraud-only -- no RAG side exists
        # yet for that one).
        runnable = {d.id for d in loader.list_domains(runnable_only=True)}
        assert runnable == ALL_DOMAIN_IDS

    def test_no_stub_domains_remain(self, loader):
        for domain_id in ALL_DOMAIN_IDS:
            config = loader.get_domain(domain_id)
            assert config.pipeline is not None
            assert config.is_runnable

    def test_unknown_domain_raises(self, loader):
        with pytest.raises(KeyError):
            loader.get_domain("does_not_exist")

    def test_env_interpolation_resolves(self, monkeypatch):
        # Fresh loader, not the shared module-scoped fixture -- reload()-ing
        # the shared one would leak env-dependent state into other tests.
        monkeypatch.setenv("RISK_THRESHOLD", "12345")
        config = ConfigLoader().get_domain("banking")
        assert config.extra["thresholds"]["risk_amount"] == "12345"

    def test_env_interpolation_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        config = ConfigLoader().get_domain("banking")
        assert config.extra["model"]["name"] == "gpt-4o-mini"

    def test_classification_hints_present_for_working_domains(self, loader):
        for domain_id in ALL_DOMAIN_IDS:
            config = loader.get_domain(domain_id)
            assert len(config.classification_hints.keywords) > 0

    def test_document_domains_declare_capabilities(self, loader):
        # All 4 domains are now unified (one pipeline, fraud detection
        # AND document chat) and DO declare capabilities -- see
        # core/config_loader.py's CapabilityConfig docstring for why this
        # distinction exists.
        for domain_id, auto_id in (
            ("banking", "statement_review"),
            ("payroll_hr", "payroll_audit"),
            ("insurance", "claim_review"),
            ("financial_services", "statement_review"),
        ):
            config = loader.get_domain(domain_id)
            ids = {c.id for c in config.capabilities}
            assert ids == {"document_qa", auto_id}
            assert config.auto_capability is not None
            assert config.auto_capability.id == auto_id
            assert config.auto_capability.query  # non-empty fixed query


class TestAgentOrchestrator:
    # No stub domains remain to test DomainNotRunnableError against for
    # real (all 4 configs are status: working now) -- covered instead by
    # test_unknown_domain_raises (KeyError, a different failure mode) and
    # by config_loader's own is_runnable property tests above.

    def test_list_domains_matches_config_loader(self):
        orchestrator = AgentOrchestrator()
        assert {d.id for d in orchestrator.list_domains()} == ALL_DOMAIN_IDS


class TestDomainClassifierAgent:
    def test_fraud_keywords_route_to_banking(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Suspicious card transaction flagged for chargeback review, amount $500, merchant unknown."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "banking"

    def test_statement_keywords_also_route_to_banking(self):
        # Banking's document side lives under the same domain id as its
        # fraud detection -- statement-shaped text routes here too, not
        # to a second id.
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Bank Statement — Account Number ACC001, Opening Balance $500, Closing Balance $1200, "
            "IBAN GB29NWBK60161331926819, withdrawal on 2024-01-05."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "banking"

    def test_all_four_working_domains_are_candidates(self):
        agent = DomainClassifierAgent()
        result = agent.classify("some random text")
        ids = {g.domain_id for g in result.all_scores}
        assert ids == ALL_DOMAIN_IDS

    def test_payslip_keywords_route_to_payroll_hr(self):
        # Payroll's document side lives under the same payroll_hr domain
        # id as its fraud detection -- payslip-shaped text routes here too.
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Payslip for pay period March 1-14: gross pay $3,062.50, federal "
            "withholding $367.50, 401k deduction, net pay $2,450.00."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "payroll_hr"

    def test_claim_keywords_route_to_insurance(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Insurance claim for reimbursement — patient diagnosis code I25.10, "
            "claim amount $500, submitted to Medicaid."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "insurance"

    def test_job_posting_keywords_route_to_payroll_hr(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Job posting: Data Entry Specialist, apply now, no experience "
            "needed, work from home opportunity, contact recruiter."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "payroll_hr"

    def test_fintech_keywords_route_to_financial_services(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Blockchain exchange alert: possible wash trading and pump "
            "and dump activity detected on a cross-chain DEX token "
            "transaction, gas fee unusually high."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "financial_services"

    def test_payroll_register_keywords_route_to_payroll_hr(self):
        # Prose, not a literal CSV header dump -- exercises the keyword path,
        # not the CSV_SCHEMA_HINTS path covered by TestCsvSchemaClassification
        # below.
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Payroll register export for the recruiter's new-hire cost "
            "review: base pay, overtime pay, gross pay, and deduction "
            "rates across every candidate hired this quarter."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "payroll_hr"


class TestCsvSchemaClassification:
    """A raw CSV header row has no prose for keyword matching to latch
    onto (e.g. "V1,V2,...,V28,Amount,Class"), so DomainClassifierAgent
    matches it against each domain's known Tier1/Tier2 column schemas
    instead -- see CSV_SCHEMA_HINTS in core/domain_classifier_agent.py."""

    def test_banking_tier1_header_routes_to_banking(self):
        agent = DomainClassifierAgent()
        header = ",".join([f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"])
        result = agent.classify(header)
        assert result.method == "csv_schema"
        assert result.best_guess.domain_id == "banking"

    def test_insurance_tier1_header_routes_to_insurance(self):
        agent = DomainClassifierAgent()
        result = agent.classify("Claim_Amount,Approved_Amount,Days_Between_Service_and_Claim,Provider_ID")
        assert result.method == "csv_schema"
        assert result.best_guess.domain_id == "insurance"

    def test_financial_services_tier1_header_routes_to_financial_services(self):
        agent = DomainClassifierAgent()
        result = agent.classify("transaction_amount_usd,sender_wallet_age_days,is_cross_chain,receiver_wallet_id")
        assert result.method == "csv_schema"
        assert result.best_guess.domain_id == "financial_services"

    def test_payroll_tier1_header_routes_to_payroll_hr(self):
        agent = DomainClassifierAgent()
        result = agent.classify("BasePay,OvertimePay,OtherPay,Stated_TotalPay,Year")
        assert result.method == "csv_schema"
        assert result.best_guess.domain_id == "payroll_hr"

    def test_hr_posting_tier1_header_routes_to_payroll_hr(self):
        agent = DomainClassifierAgent()
        result = agent.classify("title,description,telecommuting,has_company_logo,has_questions")
        assert result.method == "csv_schema"
        assert result.best_guess.domain_id == "payroll_hr"

    def test_unrecognized_csv_header_falls_back_to_keyword_path(self):
        # No domain's schema matches this -- must degrade to the keyword
        # path (score 0 / no confident guess) rather than force a wrong tier.
        # Not asserting an exact method string: an all-zero keyword tie is
        # "ambiguous" by definition (margin 0.0 < AMBIGUITY_MARGIN), so
        # whether this resolves as "keyword" or "keyword+llm" legitimately
        # depends on whether OPENAI_API_KEY is set in the environment --
        # both are correct outcomes of *not* taking the csv_schema path.
        agent = DomainClassifierAgent()
        result = agent.classify("widget_id,shelf_location,quantity_on_hand")
        assert result.method != "csv_schema"

    def test_prose_containing_a_comma_is_not_mistaken_for_a_csv_header(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Suspicious $500 card transaction at 2am, chargeback flagged."
        )
        assert result.method == "keyword"
