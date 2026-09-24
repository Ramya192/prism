# ui/guard.py — the public deployment has no login, and every ingest,
# chat turn and manual-entry score spends the operator's OpenAI key. This
# is the cheap protection that fits a single-process Streamlit app:
#
#   - per-session sliding-window rate limits, so one visitor (or a script
#     hammering the page) can't run up the bill alone, and
#   - global daily budgets across all visitors, so the worst case for a day
#     is bounded no matter how many sessions show up, and
#   - an optional shared access code (PRISM_ACCESS_CODE) that gates the
#     whole app when set -- off by default so a portfolio demo stays open.
#
# It is in-process state: it resets on restart and isn't shared across
# workers, which is right for this single-container deployment. It is a
# cost guard, not authentication -- it does not make the site private.
# Also set a monthly spend cap in the OpenAI dashboard as the hard stop.
#
# Every limit is an env var; 0 disables that limit.
#   PRISM_LIMIT_INGEST_PER_HOUR  (per session, default 20)
#   PRISM_LIMIT_CHAT_PER_HOUR    (per session, default 60)
#   PRISM_LIMIT_SCORE_PER_HOUR   (per session, default 60)
#   PRISM_DAILY_INGEST_BUDGET    (all sessions, default 300)
#   PRISM_DAILY_CHAT_BUDGET      (all sessions, default 1500)
#   PRISM_DAILY_SCORE_BUDGET     (all sessions, default 1500)

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import defaultdict, deque

import streamlit as st

from core.rag.base_document_loader_agent import DuplicateDocumentError, EmptyDocumentError

HOUR, DAY = 3600, 86400
SWEEP_EVERY = 300   # seconds between sweeps of idle-session counters

_DEFAULTS = {
    "ingest": (20, 300),
    "chat": (60, 1500),
    "score": (60, 1500),
}
_LABELS = {"ingest": "document uploads", "chat": "chat messages", "score": "manual scores"}


def _limit(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, default))
    except ValueError:
        return default


class RateLimiter:
    """Sliding-window counters keyed by (action, key). Thread-safe -- the
    Streamlit server runs each session's script in its own thread."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def _count(self, key: tuple[str, str], window: int, now: float) -> int:
        hits = self._hits.get(key)
        if hits is None:
            return 0
        while hits and hits[0] <= now - window:
            hits.popleft()
        if not hits:
            del self._hits[key]
        return len(hits)

    def _sweep(self, now: float) -> None:
        """Drops counters for sessions that have gone quiet. _count() only
        prunes a key when that same key is checked again, and a visitor who
        never returns never is -- without this the table would grow with
        every session for the life of the process. Caller holds the lock."""
        if now - self._last_sweep < SWEEP_EVERY:
            return
        self._last_sweep = now
        for key in list(self._hits):
            self._count(key, DAY if key[1] == "*" else HOUR, now)

    def check(self, action: str, session: str) -> str | None:
        """Counts one `action` for `session` and returns None if allowed,
        or a user-facing reason if a limit is hit (in which case nothing is
        counted)."""
        per_session = _limit(f"PRISM_LIMIT_{action.upper()}_PER_HOUR", _DEFAULTS[action][0])
        daily = _limit(f"PRISM_DAILY_{action.upper()}_BUDGET", _DEFAULTS[action][1])
        now = self._clock()
        with self._lock:
            self._sweep(now)
            if per_session and self._count((action, session), HOUR, now) >= per_session:
                return (f"Rate limit reached: {per_session} {_LABELS[action]} per hour per session. "
                        "Please try again later.")
            if daily and self._count((action, "*"), DAY, now) >= daily:
                return (f"This demo's daily budget of {daily} {_LABELS[action]} has been used up. "
                        "Please come back tomorrow.")
            self._hits[(action, session)].append(now)
            self._hits[(action, "*")].append(now)
        return None


_LIMITER = RateLimiter()


def get_owner() -> str:
    """A stable per-browser-session id. Namespaces this visitor's stored
    documents (core/rag/base_document_loader_agent.stored_name()) so they
    can't overwrite or be deduplicated against another visitor's."""
    if "_owner_id" not in st.session_state:
        st.session_state["_owner_id"] = uuid.uuid4().hex[:8]
    return st.session_state["_owner_id"]


def allow(action: str) -> bool:
    """Counts one action against the current session and the daily budget;
    shows the reason and returns False if it's over a limit."""
    reason = _LIMITER.check(action, get_owner())
    if reason:
        st.warning(reason)
        return False
    return True


def require_access_code() -> None:
    """If PRISM_ACCESS_CODE is set, blocks the rest of the page (st.stop)
    until the visitor enters it. No-op when unset."""
    code = os.getenv("PRISM_ACCESS_CODE")
    if not code or st.session_state.get("_access_ok"):
        return
    entered = st.text_input("Access code", type="password", key="_access_code_input")
    if entered and entered == code:
        st.session_state["_access_ok"] = True
        st.rerun()
    if entered:
        st.error("Incorrect access code.")
    st.stop()


def describe_ingest_error(e: Exception) -> str:
    """A one-line message for anything ingest() can raise, so the UI never
    shows a raw traceback. Known cases get a specific explanation; the
    rest get a generic one (the full exception is logged by the caller)."""
    if isinstance(e, DuplicateDocumentError):
        return f"Duplicate upload — {e}"
    if isinstance(e, EmptyDocumentError):
        return str(e)
    name = type(e).__name__
    if "RateLimit" in name or "APIConnection" in name or "Timeout" in name or "Authentication" in name:
        return ("The AI service couldn't be reached or rejected the request "
                f"({name}). Please try again in a moment.")
    return (f"Couldn't process this file ({name}). It may be corrupt, password-protected, "
            "or not the kind of document this domain handles.")
