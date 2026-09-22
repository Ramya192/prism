# core/config_loader.py
# ConfigLoader — reads every configs/*.yaml, validates the shared "domain"
# envelope with Pydantic, and resolves ${VAR:default} placeholders the same
# way both source projects' settings.py files already did with
# os.getenv(name, default) — just done once, centrally, at load time.
#
# Deliberately does NOT force domain-specific sections (thresholds, model,
# vector_db, llm, data, eval) into one shared schema: bfsi_fraud and
# bfsi_documents genuinely need different config shapes. Those sections are
# kept as a resolved raw dict on DomainConfig.extra, and each domain's own
# Pipeline class (see domains/<id>/pipeline.py) is what knows how to read it.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _resolve_env(value):
    """Recursively replace ${VAR:default} in strings/dicts/lists with the
    environment value, falling back to the given default (or "" if none
    and the env var is unset). Mirrors os.getenv(VAR, default)."""
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            var_name, default = match.group(1), match.group(2)
            return os.getenv(var_name, default if default is not None else "")
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class AgentConfig(BaseModel):
    id: str
    class_path: str = Field(alias="class")

    model_config = {"populate_by_name": True}


class PipelineConfig(BaseModel):
    module: str
    class_name: str = Field(alias="class")

    model_config = {"populate_by_name": True}


class ClassificationHints(BaseModel):
    keywords: list[str] = Field(default_factory=list)


class CapabilityConfig(BaseModel):
    """One analysis a document-shaped domain can run against an ingested
    file. "auto" capabilities run immediately inside Pipeline.ingest() —
    the one-upload verdict (e.g. bfsi_documents' anomaly_scan, payroll's
    payroll_audit). "on_demand" capabilities (document_qa) are the open
    chat step the user can keep using afterward, against the same
    ingestion — never a reason to re-upload the same document."""

    id: str
    label: str
    trigger: Literal["auto", "on_demand"] = "on_demand"
    query: Optional[str] = None   # fixed internal query this capability runs, if any


class DomainConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    status: str = "coming_soon"   # "working" | "coming_soon"
    input_type: str = "document"

    pipeline: Optional[PipelineConfig] = None
    agents: list[AgentConfig] = Field(default_factory=list)
    classification_hints: ClassificationHints = Field(default_factory=ClassificationHints)
    # Empty for domains with no document/chat concept (e.g. bfsi_fraud's
    # one-shot transaction verdict) — capabilities only apply to
    # document-shaped domains that ingest a file once and then answer
    # more than one question about it.
    capabilities: list[CapabilityConfig] = Field(default_factory=list)

    # Everything else in the YAML (thresholds, model, vector_db, llm, data,
    # eval, ...) — domain-specific, resolved but otherwise untouched.
    extra: dict = Field(default_factory=dict)

    @property
    def is_runnable(self) -> bool:
        return self.status == "working" and self.pipeline is not None

    @property
    def auto_capability(self) -> Optional[CapabilityConfig]:
        """The capability (if any) that should run automatically right
        after ingest, before the user asks anything."""
        return next((c for c in self.capabilities if c.trigger == "auto"), None)


_KNOWN_TOP_KEYS = {"domain", "pipeline", "agents", "classification_hints", "capabilities"}


class ConfigLoader:
    """Loads and validates configs/*.yaml. Construct once, reuse — YAML is
    only read from disk on load()/reload()."""

    def __init__(self, configs_dir: str | Path = DEFAULT_CONFIGS_DIR):
        self.configs_dir = Path(configs_dir)
        self._domains: dict[str, DomainConfig] = {}
        self.load()

    def load(self) -> None:
        self._domains = {}
        for path in sorted(self.configs_dir.glob("*.yaml")):
            domain = self._load_one(path)
            self._domains[domain.id] = domain

    reload = load  # explicit alias — reload() reads the exact same files load() did

    def _load_one(self, path: Path) -> DomainConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = _resolve_env(raw)

        domain_section = raw.get("domain")
        if not domain_section:
            raise ValueError(f"{path}: missing required top-level 'domain' section")

        extra = {k: v for k, v in raw.items() if k not in _KNOWN_TOP_KEYS}

        return DomainConfig(
            **domain_section,
            pipeline=raw.get("pipeline"),
            agents=raw.get("agents", []),
            classification_hints=raw.get("classification_hints", {}),
            capabilities=raw.get("capabilities", []),
            extra=extra,
        )

    def get_domain(self, domain_id: str) -> DomainConfig:
        try:
            return self._domains[domain_id]
        except KeyError:
            raise KeyError(
                f"Unknown domain '{domain_id}'. Known domains: {sorted(self._domains)}"
            ) from None

    def list_domains(self, runnable_only: bool = False) -> list[DomainConfig]:
        domains = list(self._domains.values())
        if runnable_only:
            domains = [d for d in domains if d.is_runnable]
        return sorted(domains, key=lambda d: d.id)
