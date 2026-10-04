# core/logging_setup.py
# One place that decides where Prism's log lines go. The scorers, drift
# detectors and detector agents used to print() their status ("Trained on N
# rows", "train.csv not found. Scorer disabled") -- invisible unless you
# happened to watch stdout, and impossible to filter. They now log through
# the standard `logging` module, and entry points (streamlit_app.py,
# main.py) call configure_logging() once so those lines show up in
# `docker compose logs` and the terminal, with a level: INFO for progress,
# WARNING for an untrained/disabled scorer, ERROR for a failure.
#
# PRISM_LOG_LEVEL (default INFO) turns it up or down, e.g. WARNING to see
# only problems.

import logging
import os


def configure_logging() -> None:
    """Idempotent: Streamlit re-runs the entry script on every interaction,
    and basicConfig is a no-op once the root logger has a handler."""
    level = os.getenv("PRISM_LOG_LEVEL", "INFO").upper()
    if not isinstance(logging.getLevelName(level), int):
        level = "INFO"
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
