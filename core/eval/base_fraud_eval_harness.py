# core/eval/base_fraud_eval_harness.py
# Shared engine behind every domain's eval_harness.py — ground-truth
# evaluation of the DEPLOYED rules -> ML decision layers (NOT just the raw
# classifier; see each domain's evaluate.py/prepare_data.py for that,
# still per-domain since it's about choosing FRAUD_THRESHOLD/
# LEGIT_THRESHOLD in the first place, not about the layered system).
#
# Pulled out once every domain's DetectorAgent had independently settled
# on the identical shape banking/fraud/eval_harness.py pioneered:
#   - rule_based_filter(record, ...) -> a "Risk Level / Reason / Action"
#     string (every real rule branch is a BLOCK) or None ("no rule fired").
#   - a scorer with .score_batch(df), .FRAUD_THRESHOLD, .LEGIT_THRESHOLD.
# Reusing this logic across domains means banking's own carefully-verified
# behavior (rule layer always resolves as BLOCK, ML layer resolves only
# past either threshold, everything else escapes to the LLM) is the
# actual behavior tested everywhere, not a re-derived approximation.
#
# The always-run part (run_tier_eval) deliberately does NOT call the LLM,
# for the same reason banking's original didn't: the point of a
# ground-truth HARNESS (vs. a one-off manual test) is being able to re-run
# it after every change for free.
#
# run_llm_sample_eval/LlmSampleSpec below are the generalized form of the
# OPTIONAL extension in banking/fraud/eval_harness.py, which keeps its
# own separate hand-written implementation rather than using this shared
# one -- a small stratified sample that DOES call the real LLM, gated
# behind --with-llm-sample and OPENAI_API_KEY, reported separately from
# the free deterministic numbers above rather than folded into them
# (small sample, seed-dependent).

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class TierSpec:
    """One fraud tier's evaluation inputs. `rule_fn` should already be
    bound to this tier (e.g. `functools.partial(DetectorAgent.rule_based_filter,
    tier="tier2")`, or the bare staticmethod for a domain with only one
    tier) — this module doesn't know or care about a domain's own
    tier-selection signature, only that calling it with one row-dict
    returns a verdict string or None."""

    name: str  # e.g. "Tier 1 — Claim_Amount schema"
    holdout_path: str
    label_col: str
    rule_fn: Callable[[dict], "str | None"]
    scorer: Any  # needs .score_batch(df), .FRAUD_THRESHOLD, .LEGIT_THRESHOLD
    positive_label: Any = 1  # value in label_col meaning "fraud"


def _is_block(verdict: "str | None") -> "bool | None":
    """None = rule didn't fire (unresolved). True/False = it fired and
    said BLOCK/APPROVE. A malformed verdict (no Action: line) is treated
    as unresolved rather than guessed, same conservative default as
    "rule didn't fire"."""
    if verdict is None:
        return None
    for line in verdict.splitlines():
        if line.strip().lower().startswith("action"):
            return "block" in line.lower()
    return None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _empty_metrics() -> dict:
    return {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def run_tier_eval(tier: TierSpec) -> dict:
    """Deterministic (rule + ML) evaluation for a single tier's holdout —
    same two-layer logic as banking/fraud/eval_harness.py's original
    run_deterministic_eval(), generalized to any TierSpec."""
    df = pd.read_csv(tier.holdout_path)
    y = (df[tier.label_col] == tier.positive_label).to_numpy().astype(int)

    scores = np.asarray(tier.scorer.score_batch(df))

    resolved = np.zeros(len(df), dtype=bool)
    predicted_fraud = np.zeros(len(df), dtype=bool)
    for i, row in enumerate(df.to_dict("records")):
        is_block = _is_block(tier.rule_fn(row))
        if is_block is not None:
            resolved[i] = True
            predicted_fraud[i] = is_block
    rule_resolved = resolved.copy()
    rule_predicted = predicted_fraud.copy()

    ml_resolved = (~resolved) & ((scores >= tier.scorer.FRAUD_THRESHOLD) | (scores <= tier.scorer.LEGIT_THRESHOLD))
    ml_predicted = ml_resolved & (scores >= tier.scorer.FRAUD_THRESHOLD)

    combined_resolved = resolved | ml_resolved
    combined_predicted = np.where(resolved, predicted_fraud, ml_predicted)

    escaped = ~combined_resolved

    return {
        "tier_name": tier.name,
        "holdout_size": len(df),
        "base_fraud_rate": float(y.mean()) if len(df) else 0.0,
        "rule_layer": {
            "resolved_count": int(rule_resolved.sum()),
            **(_metrics(y[rule_resolved], rule_predicted[rule_resolved]) if rule_resolved.any() else _empty_metrics()),
        },
        "ml_layer": {
            "resolved_count": int(ml_resolved.sum()),
            **(_metrics(y[ml_resolved], ml_predicted[ml_resolved]) if ml_resolved.any() else _empty_metrics()),
        },
        "combined_deterministic": {
            "coverage_pct": float(combined_resolved.mean() * 100) if len(df) else 0.0,
            **(_metrics(y[combined_resolved], combined_predicted[combined_resolved]) if combined_resolved.any() else _empty_metrics()),
        },
        "escaped_to_llm": {
            "count": int(escaped.sum()),
            "pct_of_holdout": float(escaped.mean() * 100) if len(df) else 0.0,
            "fraud_among_escaped": int(y[escaped].sum()) if escaped.any() else 0,
            "fraud_rate_among_escaped": float(y[escaped].mean()) if escaped.any() else 0.0,
        },
    }


def render_markdown(domain_title: str, source_note: str, tier_results: list[dict]) -> str:
    """`source_note` is a one-line attribution for where the numbers came
    from (e.g. which holdout files), rendered right under the title."""
    lines = [
        f"# Evaluation Results — {domain_title}",
        "",
        source_note,
        "",
        "Deterministic layers only (rule-based pre-filter + ML fast-path) — no LLM calls in this "
        "report; see core/eval/base_fraud_eval_harness.py's module docstring for why.",
        "",
    ]
    for d in tier_results:
        lines += [
            f"## {d['tier_name']}",
            "",
            f"- Holdout size: **{d['holdout_size']:,}** rows, **{d['base_fraud_rate']:.4%}** real fraud rate",
            f"- **{d['combined_deterministic']['coverage_pct']:.1f}%** of rows resolved without an LLM call",
            "",
            "| Layer | Resolved | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
            f"| Rule-based pre-filter | {d['rule_layer']['resolved_count']:,} | "
            f"{d['rule_layer']['precision']:.4f} | {d['rule_layer']['recall']:.4f} | {d['rule_layer']['f1']:.4f} |",
            f"| ML fast-path | {d['ml_layer']['resolved_count']:,} | "
            f"{d['ml_layer']['precision']:.4f} | {d['ml_layer']['recall']:.4f} | {d['ml_layer']['f1']:.4f} |",
            f"| **Combined (deterministic only)** | {d['combined_deterministic']['n']:,} | "
            f"**{d['combined_deterministic']['precision']:.4f}** | **{d['combined_deterministic']['recall']:.4f}** | "
            f"**{d['combined_deterministic']['f1']:.4f}** |",
            "",
            f"Confusion matrix (combined): TP={d['combined_deterministic']['tp']}, "
            f"FP={d['combined_deterministic']['fp']}, FN={d['combined_deterministic']['fn']}, "
            f"TN={d['combined_deterministic']['tn']}",
            "",
            f"- **{d['escaped_to_llm']['count']:,}** rows ({d['escaped_to_llm']['pct_of_holdout']:.1f}% of this "
            "holdout) escape to the LLM layer — genuinely borderline for both layers.",
            f"- Of those, **{d['escaped_to_llm']['fraud_among_escaped']}** are actually fraud "
            f"({d['escaped_to_llm']['fraud_rate_among_escaped']:.2%} of the escaped set).",
            "",
        ]
    return "\n".join(lines)


_VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def _llm_verdict_is_fraud(raw: str) -> bool:
    """Every domain's detector agent implements parse_response()+is_fraud()
    identically: pull the "Risk Level: <LOW/MEDIUM/HIGH>" line out of the
    LLM's free-form response, defaulting to UNKNOWN if missing/malformed,
    and treat MEDIUM/HIGH/UNKNOWN as fraud (only a clean LOW is not).
    Reproduced here directly rather than importing one domain's detector
    class just for these two static methods."""
    risk_level = "UNKNOWN"
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("risk level"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                value = parts[1].strip().upper()
                first_word = value.split()[0].rstrip(".,;") if value else ""
                risk_level = first_word if first_word in _VALID_RISK_LEVELS else "UNKNOWN"
            break
    return risk_level in {"MEDIUM", "HIGH", "UNKNOWN"}


@dataclass
class LlmSampleSpec:
    """One tier's inputs for the optional real-LLM stratified-sample
    check. `rule_fn`/`ml_filter_fn`/`build_prompt_fn` should already be
    bound to this tier (functools.partial with tier=... for a detector
    agent whose methods take one, same convention as TierSpec.rule_fn;
    bare for a single-tier domain like HR). `scorer` needs `.score(record)`
    (singular, not score_batch) — reuse the same scorer instance the
    detector agent's own ml_filter dispatches to for this tier (e.g.
    agent.scorer / agent.scorer_generalizable), not a freshly-loaded one,
    so this doesn't double the ML-model load time. `llm` is the detector
    agent's own ChatOpenAI instance (agent.llm)."""

    tier_name: str
    holdout_path: str
    label_col: str
    rule_fn: Callable[[dict], "str | None"]
    ml_filter_fn: Callable[[dict], "str | None"]
    build_prompt_fn: Callable[[dict, "str | None", "float | None"], str]
    scorer: Any
    llm: Any
    system_message: str
    positive_label: Any = 1
    sample_size: int = 150
    fraud_fraction: float = 0.3  # oversample the positive class — most of these
    # domains' real base rates are well under fraud_fraction, and a uniform
    # random sample this size would likely contain few or zero positives.


def run_llm_sample_eval(spec: LlmSampleSpec, seed: int = 42) -> dict | None:
    """A small, honestly-labeled real-LLM check for one tier — NOT part of
    the free, regression-tested numbers run_tier_eval() produces. Skipped
    with a clear message if OPENAI_API_KEY isn't set. Mirrors banking/
    fraud/eval_harness.py's original run_llm_sample_eval exactly (stratified
    sample, rule -> ML -> real LLM call in that order, LLM failures excluded
    from metrics rather than disguised as a verdict) generalized to any
    domain's TierSpec-shaped detector agent."""
    if not os.getenv("OPENAI_API_KEY"):
        print(f"  [eval_harness] OPENAI_API_KEY not set — skipping the LLM-sample check for {spec.tier_name}.")
        return None

    from langchain_core.messages import HumanMessage, SystemMessage

    df = pd.read_csv(spec.holdout_path)
    is_positive = df[spec.label_col] == spec.positive_label

    positive_df = df[is_positive]
    negative_df = df[~is_positive]
    n_positive = min(len(positive_df), int(spec.sample_size * spec.fraud_fraction))
    n_negative = min(len(negative_df), spec.sample_size - n_positive)
    if n_positive + n_negative == 0:
        print(f"  [eval_harness] Empty holdout for {spec.tier_name} — skipping the LLM-sample check.")
        return None

    parts = []
    if n_positive:
        parts.append(positive_df.sample(n=n_positive, random_state=seed))
    if n_negative:
        parts.append(negative_df.sample(n=n_negative, random_state=seed))
    # shuffle so layer order isn't correlated with row order
    sample = pd.concat(parts).sample(frac=1, random_state=seed)

    print(f"  [eval_harness] Running the full 3-layer pipeline on {len(sample)} sampled "
          f"records for {spec.tier_name} ({n_positive} positive, {n_negative} negative)...")

    y_true, y_pred = [], []
    layer_hit = {"rule": 0, "ml": 0, "llm": 0}
    llm_failures = 0
    start = time.perf_counter()

    for row in sample.to_dict("records"):
        rule_verdict = spec.rule_fn(row)
        if rule_verdict is not None:
            layer_hit["rule"] += 1
            pred = _is_block(rule_verdict)
        else:
            ml_verdict = spec.ml_filter_fn(row)
            if ml_verdict is not None:
                layer_hit["ml"] += 1
                pred = _is_block(ml_verdict)
            else:
                # Deliberately calling the LLM directly here rather than a
                # convenience analyse()-style method that might swallow API
                # errors internally and return a canned fallback verdict
                # indistinguishable from a real one — see banking/fraud/
                # eval_harness.py's original run_llm_sample_eval docstring
                # for the real bug this avoids.
                ml_score = spec.scorer.score(row)
                try:
                    messages = [
                        SystemMessage(content=spec.system_message),
                        HumanMessage(content=spec.build_prompt_fn(row, None, ml_score)),
                    ]
                    raw = spec.llm.invoke(messages).content
                    layer_hit["llm"] += 1
                    pred = _llm_verdict_is_fraud(raw)
                except Exception as e:
                    llm_failures += 1
                    print(f"  [eval_harness] LLM call failed, excluding this row from metrics: {e}")
                    continue  # excluded from y_true/y_pred entirely — not counted either way
        y_true.append(bool(row[spec.label_col] == spec.positive_label))
        y_pred.append(pred)

    elapsed_s = time.perf_counter() - start

    if llm_failures and layer_hit["llm"] == 0:
        # Every attempted LLM call in this sample failed — there is
        # nothing honest left to report for this section at all.
        print(f"  [eval_harness] All {llm_failures} LLM call(s) for {spec.tier_name} failed "
              "(likely a bad/expired OPENAI_API_KEY) — nothing real to report.")
        return {"tier_name": spec.tier_name, "sample_size": len(sample), "llm_failures": llm_failures, "all_failed": True}

    return {
        "tier_name": spec.tier_name,
        "sample_size": len(sample),
        "llm_failures": llm_failures,
        "all_failed": False,
        "layer_hit_counts": layer_hit,
        "elapsed_seconds": round(elapsed_s, 1),
        "fraud_fraction_cfg": spec.fraud_fraction,
        **_metrics(np.array(y_true, dtype=int), np.array(y_pred, dtype=int)),
    }


def render_llm_sample_markdown(base_fraud_rate: float, llm_sample: dict | None, tier_name: str | None = None) -> str:
    """One tier's LLM-sample section. `tier_name` is only needed when
    `llm_sample` is None (the skipped-entirely case, which carries no tier
    name of its own) — otherwise it's read off `llm_sample['tier_name']`."""
    name = tier_name if llm_sample is None else llm_sample["tier_name"]
    lines: list[str] = []

    if llm_sample is not None and llm_sample.get("all_failed"):
        lines += [
            f"### {name} — full 3-layer pipeline, stratified sample, real LLM calls",
            "",
            f"⚠️ **All {llm_sample['llm_failures']} LLM call(s) in this sample failed** — nothing genuine "
            "to report here. This almost always means `OPENAI_API_KEY` in `.env` is missing, expired, or "
            "invalid. Fix the key and re-run with `--with-llm-sample`.",
            "",
        ]
    elif llm_sample is not None:
        lh = llm_sample["layer_hit_counts"]
        failure_note = ""
        if llm_sample.get("llm_failures"):
            failure_note = (
                f" ⚠️ {llm_sample['llm_failures']} additional LLM call(s) in this sample failed and were "
                "excluded from the metrics below (not counted as correct or incorrect)."
            )
        lines += [
            f"### {name} — full 3-layer pipeline, stratified sample, real LLM calls",
            "",
            f"A {llm_sample['sample_size']}-row sample, oversampled for the positive class "
            f"({int(llm_sample['fraud_fraction_cfg'] * 100)}% vs. this holdout's real "
            f"{base_fraud_rate:.2%} — a uniform random sample this size would very likely contain few or "
            "zero positive rows and tell you nothing). This is the only number in this section that "
            "reflects the LLM's actual borderline judgment, not just the deterministic layers "
            "above." + failure_note,
            "",
            f"- Layer hits: rules={lh['rule']}, ML fast-path={lh['ml']}, **LLM={lh['llm']}**",
            f"- Wall time: {llm_sample['elapsed_seconds']}s for {llm_sample['sample_size']} records",
            f"- **Precision: {llm_sample['precision']:.4f} · Recall: {llm_sample['recall']:.4f} · "
            f"F1: {llm_sample['f1']:.4f}**",
            f"- Confusion matrix: TP={llm_sample['tp']}, FP={llm_sample['fp']}, FN={llm_sample['fn']}, "
            f"TN={llm_sample['tn']}",
            "",
            "Not folded into the deterministic numbers above on purpose — this sample is small and its "
            "exact composition depends on the random seed; treat it as a sanity check that Layer 3 is "
            "actually doing useful work, not a headline metric.",
            "",
        ]
    else:
        lines += [
            f"### {name} — full 3-layer pipeline (with real LLM calls)",
            "",
            "Skipped — set `OPENAI_API_KEY` and re-run with `--with-llm-sample` to include a stratified "
            "sample that exercises the actual LLM borderline-case reasoning, not just the two "
            "deterministic layers above.",
            "",
        ]
    return "\n".join(lines)
