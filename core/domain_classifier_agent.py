# core/domain_classifier_agent.py
# DomainClassifierAgent — the box at the top of Prism's architecture diagram:
# guesses which domain an uploaded input belongs to, but never runs a
# pipeline on its own say-so. streamlit_app.py always shows this guess to a
# human and requires an explicit confirm/override before AgentOrchestrator
# is asked to do anything.
#
# Two-stage guess, both stages free of any hard dependency on an LLM call:
#   1. Keyword overlap against each runnable domain's classification_hints
#      (configs/*.yaml) — deterministic, zero cost, always available.
#   2. If an OpenAI key is configured AND the keyword scores are close
#      (i.e. genuinely ambiguous), one cheap LLM call breaks the tie.
#      Skipped entirely when the top keyword score is already clear, so the
#      common case never pays for an LLM round-trip just to classify.

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from core.config_loader import ConfigLoader, DomainConfig

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class DomainGuess:
    domain_id: str
    domain_name: str
    score: float          # 0.0-1.0, normalised keyword-overlap fraction
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    best_guess: DomainGuess | None
    all_scores: list[DomainGuess]
    method: str            # "keyword" or "keyword+llm"
    llm_reasoning: str | None = None


class DomainClassifierAgent:
    # If the top two keyword scores are within this margin of each other,
    # treat it as ambiguous and ask the LLM to break the tie (only if a key
    # is configured). Otherwise the keyword winner stands on its own.
    AMBIGUITY_MARGIN = 0.15

    def __init__(self, config_loader: ConfigLoader | None = None):
        self.config_loader = config_loader or ConfigLoader()

    def _keyword_scores(self, text: str, domains: list[DomainConfig]) -> list[DomainGuess]:
        tokens = _tokenize(text)
        guesses = []
        for d in domains:
            keywords = d.classification_hints.keywords
            if not keywords:
                guesses.append(DomainGuess(d.id, d.name, 0.0, []))
                continue
            matched = [
                kw for kw in keywords
                if all(w in tokens for w in _tokenize(kw)) if _tokenize(kw)
            ]
            score = len(matched) / len(keywords)
            guesses.append(DomainGuess(d.id, d.name, score, matched))
        return sorted(guesses, key=lambda g: g.score, reverse=True)

    def _llm_tiebreak(self, text: str, candidates: list[DomainGuess]) -> tuple[str | None, str | None]:
        """Returns (chosen_domain_id, reasoning) or (None, None) if no key
        is configured / the call fails — callers fall back to the keyword
        winner either way."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, None
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            model_name = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
            options = "\n".join(f"- {c.domain_id}: {c.domain_name}" for c in candidates)
            snippet = text[:1500]
            prompt = f"""You are routing an uploaded input to the correct pipeline in a
multi-domain platform. Candidate domains:
{options}

Input excerpt:
\"\"\"{snippet}\"\"\"

Respond in exactly this format:
Domain: <one of the domain ids above>
Reason: <one sentence>"""
            response = llm.invoke(
                [SystemMessage(content="Be concise and precise."), HumanMessage(content=prompt)]
            )
            raw = response.content
            domain_id, reason = None, None
            for line in raw.splitlines():
                if line.lower().startswith("domain:"):
                    domain_id = line.split(":", 1)[1].strip()
                elif line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
            valid_ids = {c.domain_id for c in candidates}
            if domain_id not in valid_ids:
                return None, None
            return domain_id, reason
        except Exception as e:  # LLM unavailable/misconfigured — degrade to keyword result
            return None, f"LLM tiebreak unavailable: {e}"

    def classify(self, text: str, runnable_only: bool = True) -> ClassificationResult:
        domains = self.config_loader.list_domains(runnable_only=runnable_only)
        if not domains:
            return ClassificationResult(best_guess=None, all_scores=[], method="keyword")

        scored = self._keyword_scores(text or "", domains)
        top_two = scored[:2]

        ambiguous = (
            len(top_two) == 2
            and (top_two[0].score - top_two[1].score) < self.AMBIGUITY_MARGIN
        )

        if ambiguous:
            chosen_id, reasoning = self._llm_tiebreak(text, top_two)
            if chosen_id:
                chosen = next(g for g in scored if g.domain_id == chosen_id)
                return ClassificationResult(
                    best_guess=chosen, all_scores=scored,
                    method="keyword+llm", llm_reasoning=reasoning,
                )

        return ClassificationResult(best_guess=scored[0], all_scores=scored, method="keyword")
