# core/usage_budget.py -- a rolling 24-hour counter for work that spends the
# operator's API key. ui/guard.py already caps how many uploads and chat
# turns a public site accepts, but one upload can fan out into thousands of
# LLM calls (every CSV row that escapes the rules and ML tiers is one call),
# so the fan-out itself needs a budget. UnifiedDomainPipeline.ingest() draws
# scored rows from ROWS_BUDGET before it scores anything.
#
# In-process state, like ui/guard.py: it resets on restart and isn't shared
# across workers, which fits this single-container deployment. Rows are
# charged whichever tier ends up handling them -- the free rules/ML tiers
# are counted too, since which rows reach the LLM isn't known up front.

from __future__ import annotations

import os
import threading
import time
from collections import deque

DAY = 86400


class DailyBudget:
    def __init__(self, env: str, default: int, clock=time.time):
        """`env` names the variable holding the limit (read on every call,
        so it can be changed without a restart); 0 disables the budget."""
        self._env, self._default, self._clock = env, default, clock
        self._grants: deque[tuple[float, int]] = deque()   # (time, amount)
        self._used = 0
        self._lock = threading.Lock()

    def limit(self) -> int:
        try:
            return int(os.getenv(self._env, self._default))
        except ValueError:
            return self._default

    def grant(self, requested: int) -> int:
        """Reserves up to `requested` units and returns how many were
        actually granted (0..requested) -- a partial grant is possible when
        the day's budget is nearly used up."""
        limit = self.limit()
        if limit <= 0 or requested <= 0:
            return max(requested, 0)
        now = self._clock()
        with self._lock:
            while self._grants and self._grants[0][0] <= now - DAY:
                self._used -= self._grants.popleft()[1]
            granted = max(0, min(requested, limit - self._used))
            if granted:
                self._grants.append((now, granted))
                self._used += granted
        return granted


ROWS_BUDGET = DailyBudget("PRISM_DAILY_SCORED_ROWS_BUDGET", 5000)
