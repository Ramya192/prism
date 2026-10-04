# Every detector agent stamps which layer (rules / ML / LLM) decided a verdict,
# and parse_response() reads it back into parsed["decided_by"] -- so the UI
# never has to guess it from the reason text. Offline: scorers, rules and the
# LLM are all stubbed, no API key or trained model needed.

from types import SimpleNamespace

import pytest

from core.detection_layer import LLM_REASONING, ML_DETECTOR, RULE_ENGINE, parse_decided_by, tag_layer
from domains.banking.fraud.agents.detector_agent import FraudDetectorAgent
from domains.financial_services.agents.fintech_detector_agent import FintechDetectorAgent
from domains.insurance.agents.healthcare_detector_agent import HealthcareDetectorAgent
from domains.payroll_hr.hr.agents.hr_detector_agent import HrDetectorAgent
from domains.payroll_hr.payroll.agents.payroll_detector_agent import PayrollDetectorAgent

AGENTS = [FraudDetectorAgent, HealthcareDetectorAgent, PayrollDetectorAgent, HrDetectorAgent, FintechDetectorAgent]

# A short, percentage-free BLOCK: exactly what the old text heuristic mistook for a rule hit.
LLM_BLOCK = "Risk Level: HIGH\nReason: Looks like card testing.\nAction: BLOCK"
RULE_TEXT = "Risk Level: HIGH\nReason: Zero-dollar transaction.\nAction: BLOCK"
ML_TEXT = "Risk Level: HIGH\nReason: ML model fraud probability 99%.\nAction: BLOCK"


def _agent(cls, *, rule=None, ml=None, llm_text=LLM_BLOCK, llm_raises=False):
    agent = cls.__new__(cls)
    agent.cases_reviewed = 0
    agent.detect_tier = lambda *a: "tier2"              # tier2: reachable by rules/ML/LLM, no drift check
    agent.has_recognizable_schema = lambda *a: True      # HR's equivalent
    agent.rule_based_filter = lambda *a: rule
    agent.ml_filter = lambda *a: ml
    agent.build_prompt = lambda *a, **k: "prompt"
    scorer = SimpleNamespace(score=lambda record: 0.5)
    agent.scorer = agent.scorer_generalizable = agent.scorer_tier2 = scorer

    def invoke(messages):
        if llm_raises:
            raise RuntimeError("api down")
        return SimpleNamespace(content=llm_text)
    agent.llm = SimpleNamespace(invoke=invoke)
    return agent


def _decided_by(cls, **kwargs):
    response = _agent(cls, **kwargs).analyse({"Amount": 100.0, "hour": 14})
    return cls.parse_response(response)["decided_by"]


@pytest.mark.parametrize("cls", AGENTS)
def test_rule_hit_is_recorded_as_rule_engine(cls):
    assert _decided_by(cls, rule=RULE_TEXT) == RULE_ENGINE


@pytest.mark.parametrize("cls", AGENTS)
def test_ml_hit_is_recorded_as_ml_detector(cls):
    assert _decided_by(cls, ml=ML_TEXT) == ML_DETECTOR


@pytest.mark.parametrize("cls", AGENTS)
def test_llm_block_with_a_short_reason_is_still_llm_reasoning(cls):
    assert _decided_by(cls) == LLM_REASONING


@pytest.mark.parametrize("cls", AGENTS)
def test_llm_api_failure_is_recorded_as_llm_reasoning(cls):
    assert _decided_by(cls, llm_raises=True) == LLM_REASONING


@pytest.mark.parametrize("cls", AGENTS)
def test_tag_does_not_disturb_the_verdict_fields(cls):
    parsed = cls.parse_response(_agent(cls, rule=RULE_TEXT).analyse({"Amount": 0.0, "hour": 2}))
    assert (parsed["risk_level"], parsed["action"], parsed["reason"]) == ("HIGH", "BLOCK", "Zero-dollar transaction.")


@pytest.mark.parametrize("cls", AGENTS)
def test_untagged_response_has_no_deciding_layer(cls):
    assert "decided_by" not in cls.parse_response(LLM_BLOCK)


def test_helpers_round_trip_and_reject_unknown_layers():
    assert parse_decided_by(tag_layer("x", ML_DETECTOR).splitlines()[-1]) == ML_DETECTOR
    assert parse_decided_by("Decided by: a guess") is None
    assert parse_decided_by("Reason: Decided by committee") is None
