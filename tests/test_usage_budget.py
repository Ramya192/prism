# core/usage_budget.py -- the rolling daily scored-rows budget, plus its use
# in UnifiedDomainPipeline.ingest(). Offline (stubbed embedder, fake clock).

import pytest

from core.usage_budget import DAY, DailyBudget
from tests.test_duplicate_ingest import _Pipeline, _csv, loader  # noqa: F401  (loader is a fixture)


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_grants_up_to_the_limit_then_partially_then_nothing(monkeypatch):
    monkeypatch.setenv("B", "10")
    budget = DailyBudget("B", 999, Clock())
    assert budget.grant(4) == 4
    assert budget.grant(4) == 4
    assert budget.grant(4) == 2        # only 2 left today
    assert budget.grant(4) == 0


def test_budget_rolls_over_after_a_day(monkeypatch):
    monkeypatch.setenv("B", "5")
    clock = Clock()
    budget = DailyBudget("B", 999, clock)
    assert budget.grant(5) == 5
    assert budget.grant(1) == 0
    clock.now += DAY + 1
    assert budget.grant(5) == 5


def test_zero_disables_and_bad_values_fall_back_to_default(monkeypatch):
    monkeypatch.setenv("B", "0")
    assert DailyBudget("B", 3, Clock()).grant(100) == 100
    monkeypatch.setenv("B", "not-a-number")
    assert DailyBudget("B", 3, Clock()).grant(100) == 3


def test_ingest_scores_only_the_granted_rows_but_still_ingests(loader, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("B", "3")
    monkeypatch.setattr("core.unified_pipeline.ROWS_BUDGET", DailyBudget("B", 999, Clock()))
    pipeline = _Pipeline(loader)

    first = pipeline.ingest(_csv(tmp_path / "a.csv", "a,b\n1,1\n2,2\n3,3\n4,4\n5,5\n"), filename="a")
    assert first["chunks_stored"] == 5                      # ingested in full
    assert first["fraud_summary"]["total_scored"] == 3      # scored only what was granted
    assert first["fraud_summary"]["budget_limited"] is True
    assert first["fraud_summary"]["truncated"] is True

    second = pipeline.ingest(_csv(tmp_path / "b.csv", "a,b\n9,9\n8,8\n"), filename="b")
    assert second["chunks_stored"] == 2                     # still chattable
    assert second["fraud_verdicts"] == []
    assert second["fraud_summary"]["budget_limited"] is True


def test_ingest_reports_not_limited_when_budget_suffices(loader, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("B", "100")
    monkeypatch.setattr("core.unified_pipeline.ROWS_BUDGET", DailyBudget("B", 999, Clock()))
    result = _Pipeline(loader).ingest(_csv(tmp_path / "c.csv"), filename="c")
    assert result["fraud_summary"]["total_scored"] == 2
    assert "budget_limited" not in result["fraud_summary"]
