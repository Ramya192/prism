# core/orchestrator.py
# AgentOrchestrator — given a domain_id, resolves its DomainConfig via
# ConfigLoader, dynamically imports that domain's Pipeline class, and
# instantiates it once (cached per domain_id for the process lifetime —
# same idea as @st.cache_resource in both source apps' old app.py files,
# just framework-agnostic so main.py/tests/streamlit_app.py can all share it).
#
# Domains are NOT forced into one shared agent method signature. Fraud
# analysis (transaction dict in, verdict out) and document RAG (PDF in,
# ingest + query) are genuinely different shapes of work. Every domain
# Pipeline exposes at least `.run(**kwargs) -> dict`; it's free to expose
# extra methods (e.g. DocumentPipeline.ingest(...)) that callers reach via
# the instance this orchestrator hands back.

from __future__ import annotations

import importlib
from typing import Any

from core.config_loader import ConfigLoader, DomainConfig


class DomainNotRunnableError(Exception):
    """Raised when a domain is known but has no working pipeline (stub
    configs like hr_compliance.yaml / healthcare.yaml)."""


class AgentOrchestrator:
    def __init__(self, config_loader: ConfigLoader | None = None):
        self.config_loader = config_loader or ConfigLoader()
        self._pipelines: dict[str, Any] = {}

    def _instantiate_pipeline(self, config: DomainConfig):
        if not config.is_runnable:
            raise DomainNotRunnableError(
                f"Domain '{config.id}' is '{config.status}' — no pipeline configured yet."
            )
        module = importlib.import_module(config.pipeline.module)
        pipeline_cls = getattr(module, config.pipeline.class_name)
        # Every Pipeline.__init__ takes the resolved DomainConfig so it can
        # read its own thresholds/model/vector_db/... section (config.extra)
        # and instantiate the agents listed under `agents:` however it needs.
        return pipeline_cls(config)

    def get_pipeline(self, domain_id: str):
        """Returns the (cached) Pipeline instance for a domain, building it
        on first use. Raises DomainNotRunnableError for stub domains."""
        if domain_id not in self._pipelines:
            config = self.config_loader.get_domain(domain_id)
            self._pipelines[domain_id] = self._instantiate_pipeline(config)
        return self._pipelines[domain_id]

    def run(self, domain_id: str, **kwargs) -> dict:
        """Convenience wrapper for the common case — get the pipeline, run
        it. Anything needing a domain-specific extra method (e.g. ingest)
        should call get_pipeline(domain_id) directly instead."""
        return self.get_pipeline(domain_id).run(**kwargs)

    def list_domains(self, runnable_only: bool = False):
        return self.config_loader.list_domains(runnable_only=runnable_only)
