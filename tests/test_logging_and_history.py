# Offline checks for two shared-infrastructure clean-ups:
#   * scorer/detector status lines go through `logging`, not print(), and
#     core.logging_setup.configure_logging() decides where they show up;
#   * every domain's chat agent renders earlier turns via ONE shared function
#     (core/rag/chat_history.py), including the "follow-up only" note.

import logging
import re
from pathlib import Path

import pytest

from core.logging_setup import configure_logging
from core.rag.chat_history import FOLLOW_UP_ONLY_NOTE, render_history

ROOT = Path(__file__).resolve().parent.parent

HISTORY = [
    {"speaker": "assistant", "text": "Scored 768 record(s). Flagged: X"},
    {"speaker": "user", "text": "why was that flagged?"},
]


# ── chat history ──────────────────────────────────────────────────────────

def test_render_history_is_empty_without_turns():
    assert render_history(None) == "" and render_history([]) == ""


def test_render_history_lists_turns_and_says_they_are_for_follow_ups_only():
    rendered = render_history(HISTORY)
    assert "You: Scored 768 record(s). Flagged: X" in rendered
    assert "User: why was that flagged?" in rendered
    assert rendered.rstrip().endswith(FOLLOW_UP_ONLY_NOTE)


def _reasoning_agents():
    from domains.banking.documents.agents.reasoning_agent import ReasoningAgent as Banking
    from domains.financial_services.agents.reasoning_agent import ReasoningAgent as Fin
    from domains.insurance.agents.reasoning_agent import ReasoningAgent as Insurance
    from domains.payroll_hr.payroll.agents.reasoning_agent import ReasoningAgent as Payroll
    return [Banking, Fin, Insurance, Payroll]


def test_every_domain_renders_history_the_same_way():
    for agent in _reasoning_agents():
        assert agent._render_history(HISTORY) == render_history(HISTORY), agent.__module__
        assert agent._render_history(None) == ""


# ── logging ───────────────────────────────────────────────────────────────

@pytest.fixture
def clean_root_logger():
    """Restores the root logger afterwards. pytest attaches its own capture
    handlers to root around each phase, so each test empties root itself just
    before calling configure_logging()."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)


def test_configure_logging_honours_the_env_level_and_is_idempotent(clean_root_logger, monkeypatch):
    monkeypatch.setenv("PRISM_LOG_LEVEL", "warning")
    clean_root_logger.handlers.clear()
    configure_logging()
    configure_logging()
    assert clean_root_logger.level == logging.WARNING
    assert len(clean_root_logger.handlers) == 1     # a Streamlit rerun must not stack handlers


def test_configure_logging_falls_back_to_info_on_a_bad_level(clean_root_logger, monkeypatch):
    monkeypatch.setenv("PRISM_LOG_LEVEL", "loud")
    clean_root_logger.handlers.clear()
    configure_logging()
    assert clean_root_logger.level == logging.INFO


def test_no_status_print_left_in_scorers_and_detectors():
    """Every "  [Tag] message" status line is a logger call now. (CLI report
    prints -- status() methods, model_store's size table -- don't use this shape.)"""
    status_print = re.compile(r'print\(f?"  \[')
    skip = ("eval_harness", "/data/", "prepare_", "generate_", "ingest_reference_corpus",
            "audit_data", "bulk_ingest", "format_detector", "fraud/evaluate.py")
    offenders = []
    for base in ("core", "domains"):
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if any(s in rel for s in skip):
                continue
            if status_print.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)
    assert offenders == []
