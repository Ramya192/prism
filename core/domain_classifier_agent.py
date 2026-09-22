# core/domain_classifier_agent.py
# DomainClassifierAgent — the box at the top of Prism's architecture diagram:
# guesses which domain an uploaded input belongs to, but never runs a
# pipeline on its own say-so. streamlit_app.py always shows this guess to a
# human and requires an explicit confirm/override before AgentOrchestrator
# is asked to do anything.
#
# Three-stage guess, all free of any hard dependency on an LLM call:
#   0. If the input's first line looks like a CSV header row, match it
#      against each domain's known Tier1/Tier2 column schemas (mirrors the
#      REQUIRED_COLS/FEATURE_COLS each domain's scorer already checks — see
#      CSV_SCHEMA_HINTS below). A raw CSV export has no prose for keyword
#      matching to latch onto, so this is the only signal that works for a
#      bare column dump like "V1,V2,...,V28,Amount,Class". Decisive on a
#      strong match; falls through to keyword matching otherwise (e.g. a
#      CSV shaped like nothing Prism has trained on).
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

# Column-name schemas each domain's Tier1/Tier2 scorer already keys off of.
# Duplicated here (rather than imported) so the classifier stays free of
# the scorers' heavy deps (lightgbm/xgboost/sklearn) — this list only needs
# to stay in sync when a scorer's own REQUIRED_COLS/FEATURE_COLS changes.
# Sources:
#   banking:            domains/banking/fraud/tools/ml_scorer.py (FEATURE_COLS, Tier1)
#                        domains/banking/fraud/tools/generalizable_scorer.py (REQUIRED_COLS, Tier2)
#   insurance:          domains/insurance/tools/claim_scorer.py (Tier1)
#                        domains/insurance/tools/generalizable_scorer.py (Tier2)
#   financial_services: domains/financial_services/tools/scam_scorer.py (Tier1)
#                        domains/financial_services/tools/wash_trading_scorer.py (Tier2)
#   payroll_hr:         domains/payroll_hr/payroll/tools/anomaly_scorer.py (Tier1, payroll)
#                        domains/payroll_hr/payroll/tools/generalizable_scorer.py (Tier2, payroll)
#                        domains/payroll_hr/hr/tools/posting_scorer.py (Tier1, hr postings)
CSV_SCHEMA_HINTS: dict[str, list[list[str]]] = {
    "banking": [
        [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"],
        ["Transaction_Amount", "Account_Age", "Credit_Score"],
    ],
    "insurance": [
        ["Claim_Amount", "Approved_Amount", "Days_Between_Service_and_Claim"],
        ["Claim_Amount", "Policy_Number", "Provider_Patient_Distance_Miles"],
    ],
    "financial_services": [
        ["transaction_amount_usd", "sender_wallet_age_days", "is_cross_chain"],
        ["notional_usd", "round_trip_score", "counterparty_reuse_ratio"],
    ],
    "payroll_hr": [
        ["BasePay", "OvertimePay", "OtherPay", "Stated_TotalPay"],
        ["GROSS", "Deduction", "Net_Pay"],
        ["title", "description", "telecommuting", "has_company_logo", "has_questions"],
    ],
}

# A schema match needs at least this fraction of its columns present in the
# uploaded header to be treated as decisive (vs. keyword/LLM fallback).
CSV_SCHEMA_MATCH_THRESHOLD = 0.66


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _extract_csv_header(text: str) -> list[str] | None:
    """Returns the column names of `text`'s first line if it looks like a
    CSV header row (2+ comma-separated, identifier-shaped fields with no
    sentence punctuation), else None. Deliberately conservative — prose
    that happens to contain a comma must never be mistaken for a header."""
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if "," not in first_line or any(c in first_line for c in ".!?\""):
        return None
    fields = [f.strip().strip("'\"") for f in first_line.split(",")]
    if len(fields) < 2:
        return None
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", f) for f in fields):
        return None
    return fields


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

    def _csv_schema_guess(self, header_cols: list[str], domains: list[DomainConfig]) -> DomainGuess | None:
        """Best domain match for a raw CSV header row, by overlap against
        each domain's known Tier1/Tier2 schemas. Returns None if nothing
        clears CSV_SCHEMA_MATCH_THRESHOLD (e.g. a CSV shaped like nothing
        Prism has trained on) so the caller can fall back to keyword
        matching against the same header text."""
        header_lower = {c.lower() for c in header_cols}
        runnable_ids = {d.id for d in domains}
        best: tuple[float, str, list[str]] | None = None
        for domain_id, schemas in CSV_SCHEMA_HINTS.items():
            if domain_id not in runnable_ids:
                continue
            for schema in schemas:
                matched = [c for c in schema if c.lower() in header_lower]
                ratio = len(matched) / len(schema)
                if ratio >= CSV_SCHEMA_MATCH_THRESHOLD and (best is None or ratio > best[0]):
                    best = (ratio, domain_id, matched)
        if best is None:
            return None
        ratio, domain_id, matched = best
        d = next(d for d in domains if d.id == domain_id)
        return DomainGuess(d.id, d.name, ratio, matched)

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

        header_cols = _extract_csv_header(text or "")
        if header_cols:
            schema_guess = self._csv_schema_guess(header_cols, domains)
            if schema_guess:
                return ClassificationResult(best_guess=schema_guess, all_scores=[schema_guess], method="csv_schema")

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
