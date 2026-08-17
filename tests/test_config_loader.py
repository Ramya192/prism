# tests/test_config_loader.py
# New for Prism — verifies ConfigLoader/AgentOrchestrator/DomainClassifierAgent
# work against the real configs/*.yaml, and that stub domains are correctly
# excluded from anything that tries to run them.

import pytest

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator, DomainNotRunnableError
from core.domain_classifier_agent import DomainClassifierAgent


@pytest.fixture(scope="module")
def loader():
    return ConfigLoader()


class TestConfigLoader:
    def test_loads_all_four_configs(self, loader):
        ids = {d.id for d in loader.list_domains()}
        assert ids == {"bfsi_fraud", "bfsi_documents", "hr_compliance", "healthcare"}

    def test_working_domains_are_runnable(self, loader):
        runnable = {d.id for d in loader.list_domains(runnable_only=True)}
        assert runnable == {"bfsi_fraud", "bfsi_documents"}

    def test_stub_domains_have_no_pipeline(self, loader):
        for domain_id in ("hr_compliance", "healthcare"):
            config = loader.get_domain(domain_id)
            assert config.pipeline is None
            assert not config.is_runnable

    def test_unknown_domain_raises(self, loader):
        with pytest.raises(KeyError):
            loader.get_domain("does_not_exist")

    def test_env_interpolation_resolves(self, monkeypatch):
        # Fresh loader, not the shared module-scoped fixture -- reload()-ing
        # the shared one would leak env-dependent state into other tests.
        monkeypatch.setenv("RISK_THRESHOLD", "12345")
        config = ConfigLoader().get_domain("bfsi_fraud")
        assert config.extra["thresholds"]["risk_amount"] == "12345"

    def test_env_interpolation_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        config = ConfigLoader().get_domain("bfsi_fraud")
        assert config.extra["model"]["name"] == "gpt-4o-mini"

    def test_classification_hints_present_for_working_domains(self, loader):
        for domain_id in ("bfsi_fraud", "bfsi_documents"):
            config = loader.get_domain(domain_id)
            assert len(config.classification_hints.keywords) > 0


class TestAgentOrchestrator:
    def test_stub_domain_raises_not_runnable(self):
        orchestrator = AgentOrchestrator()
        with pytest.raises(DomainNotRunnableError):
            orchestrator.get_pipeline("hr_compliance")

    def test_list_domains_matches_config_loader(self):
        orchestrator = AgentOrchestrator()
        assert {d.id for d in orchestrator.list_domains()} == {
            "bfsi_fraud", "bfsi_documents", "hr_compliance", "healthcare",
        }


class TestDomainClassifierAgent:
    def test_fraud_keywords_route_to_bfsi_fraud(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Suspicious card transaction flagged for chargeback review, amount $500, merchant unknown."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "bfsi_fraud"

    def test_statement_keywords_route_to_bfsi_documents(self):
        agent = DomainClassifierAgent()
        result = agent.classify(
            "Bank Statement — Account Number ACC001, Opening Balance $500, Closing Balance $1200, "
            "IBAN GB29NWBK60161331926819, withdrawal on 2024-01-05."
        )
        assert result.best_guess is not None
        assert result.best_guess.domain_id == "bfsi_documents"

    def test_stub_domains_excluded_by_default(self):
        agent = DomainClassifierAgent()
        result = agent.classify("some random text")
        ids = {g.domain_id for g in result.all_scores}
        assert ids == {"bfsi_fraud", "bfsi_documents"}
