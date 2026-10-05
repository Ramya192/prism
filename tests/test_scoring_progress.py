"""UnifiedDomainPipeline._score_all reports progress from the calling thread
and keeps record order -- offline, no LLM calls."""
import threading

import pytest

from core.unified_pipeline import UnifiedDomainPipeline


class _Stub(UnifiedDomainPipeline):
    def __init__(self, workers):
        self.SCORING_WORKERS = workers

    def score_record(self, record, context=None):
        return {"n": record["n"]}


@pytest.mark.parametrize("workers", [1, 4])
def test_progress_counts_up_in_order_on_calling_thread(workers):
    calls, threads = [], set()

    def progress(done, total):
        calls.append((done, total))
        threads.add(threading.current_thread())

    out = _Stub(workers)._score_all([{"n": i} for i in range(10)], progress=progress)

    assert [r["n"] for r in out] == list(range(10))
    assert calls == [(i, 10) for i in range(1, 11)]
    assert threads == {threading.current_thread()}


def test_no_progress_callback_is_fine():
    assert len(_Stub(4)._score_all([{"n": i} for i in range(3)])) == 3
