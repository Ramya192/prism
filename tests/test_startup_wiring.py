# Every domain pipeline builds through the real AgentOrchestrator and comes
# up with its ML tiers actually trained -- from the committed model
# artifacts (core/model_store.py), with no train CSVs on disk. This is the
# end-to-end guard for the bug where a fresh clone (the EC2 box) silently
# ran with every scorer untrained, sending everything to the LLM tier.
#
# Offline: constructing the OpenAI/Chroma clients makes no network call, so
# a placeholder key is enough (setdefault -- a real key in .env still wins).

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-never-used")
os.environ.setdefault("MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("RISK_THRESHOLD", "50000")
os.environ.setdefault("LLM_PROVIDER", "openai")   # not ollama: no local model/reranker needed to construct

import pytest

from core.orchestrator import AgentOrchestrator


@pytest.fixture(scope="module")
def orchestrator():
    return AgentOrchestrator()


def test_banking_ml_tiers_and_drift_detector_trained(orchestrator):
    detector = orchestrator.get_pipeline("banking").fraud_pipeline.detector
    assert detector.scorer.trained
    assert detector.scorer_generalizable.trained
    assert detector.drift_detector.trained


def test_insurance_ml_tiers_trained(orchestrator):
    detector = orchestrator.get_pipeline("insurance").detector
    assert detector.scorer.trained
    assert detector.scorer_generalizable.trained


def test_financial_services_ml_tiers_trained(orchestrator):
    detector = orchestrator.get_pipeline("financial_services").detector
    assert detector.scorer.trained
    assert detector.scorer_tier2.trained


def test_payroll_hr_ml_tiers_trained(orchestrator):
    fraud = orchestrator.get_pipeline("payroll_hr").fraud_pipeline
    assert fraud.payroll_detector.scorer.trained
    assert fraud.payroll_detector.scorer_generalizable.trained
    assert fraud.hr_detector.scorer.trained
