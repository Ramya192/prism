# ui/guard.py -- rate limits, daily budgets, and friendly ingest errors.
# The RateLimiter is exercised directly with a fake clock (no Streamlit
# session needed).

import pytest

from core.rag.base_document_loader_agent import DuplicateDocumentError, EmptyDocumentError
from ui.guard import DAY, HOUR, RateLimiter, describe_ingest_error


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def test_per_session_hourly_limit(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_INGEST_PER_HOUR", "3")
    limiter = RateLimiter(clock)
    assert [limiter.check("ingest", "s1") for _ in range(3)] == [None, None, None]
    assert "3 document uploads per hour" in limiter.check("ingest", "s1")


def test_limit_is_per_session_not_global(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_INGEST_PER_HOUR", "1")
    limiter = RateLimiter(clock)
    assert limiter.check("ingest", "s1") is None
    assert limiter.check("ingest", "s1") is not None
    assert limiter.check("ingest", "s2") is None


def test_window_slides(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_CHAT_PER_HOUR", "1")
    limiter = RateLimiter(clock)
    assert limiter.check("chat", "s1") is None
    assert limiter.check("chat", "s1") is not None
    clock.now += HOUR + 1
    assert limiter.check("chat", "s1") is None


def test_daily_budget_spans_sessions(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_CHAT_PER_HOUR", "0")
    monkeypatch.setenv("PRISM_DAILY_CHAT_BUDGET", "2")
    limiter = RateLimiter(clock)
    assert limiter.check("chat", "a") is None
    assert limiter.check("chat", "b") is None
    assert "daily budget" in limiter.check("chat", "c")
    clock.now += DAY + 1
    assert limiter.check("chat", "c") is None


def test_rejected_attempts_are_not_counted(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_SCORE_PER_HOUR", "1")
    monkeypatch.setenv("PRISM_DAILY_SCORE_BUDGET", "2")
    limiter = RateLimiter(clock)
    assert limiter.check("score", "s1") is None
    for _ in range(5):
        assert limiter.check("score", "s1") is not None   # hammering s1 must not burn the daily budget
    assert limiter.check("score", "s2") is None


def test_zero_disables_a_limit(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_INGEST_PER_HOUR", "0")
    monkeypatch.setenv("PRISM_DAILY_INGEST_BUDGET", "0")
    limiter = RateLimiter(clock)
    assert all(limiter.check("ingest", "s1") is None for _ in range(200))


def test_bad_env_value_falls_back_to_default(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_INGEST_PER_HOUR", "lots")
    limiter = RateLimiter(clock)
    assert all(limiter.check("ingest", "s1") is None for _ in range(20))
    assert limiter.check("ingest", "s1") is not None   # default of 20/hour


def test_describe_ingest_error_known_and_unknown():
    assert "already ingested" in describe_ingest_error(DuplicateDocumentError("a.pdf", "a.pdf@s1"))
    assert "no extractable text" in describe_ingest_error(EmptyDocumentError("scan.pdf"))
    generic = describe_ingest_error(ValueError("boom: secret internal detail"))
    assert "ValueError" in generic and "secret internal detail" not in generic
    assert "AI service" in describe_ingest_error(type("RateLimitError", (Exception,), {})("429"))


def test_idle_sessions_are_swept_from_memory(clock, monkeypatch):
    monkeypatch.setenv("PRISM_LIMIT_CHAT_PER_HOUR", "5")
    limiter = RateLimiter(clock)
    for i in range(50):
        assert limiter.check("chat", f"visitor{i}") is None    # 50 one-off visitors
    assert len(limiter._hits) == 51                             # 50 sessions + the "*" total
    clock.now += HOUR + 1                                       # they all go quiet
    assert limiter.check("chat", "latecomer") is None           # next check sweeps
    assert set(limiter._hits) == {("chat", "latecomer"), ("chat", "*")}
