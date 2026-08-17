"""
domains/bfsi_fraud/eval_harness.py — ground-truth evaluation of the DEPLOYED
3-layer system, not just the raw ML classifier. Ported from
fraud_detection/eval_harness.py — logic unchanged, paths updated to this
domain's own data/ folder, and this is what configs/bfsi_fraud.yaml's
`eval.harness` points at.

evaluate.py already answers "how good is the RandomForest on its own" (PR-AUC,
ROC-AUC, threshold/cost sweeps) — that's classifier calibration, and it's
still the right tool for choosing FRAUD_THRESHOLD/LEGIT_THRESHOLD. This file
answers a different, harder question: "how good is the actual system a user
would hit," which means running the real rule_based_filter() and ml_filter()
decision layers — in that order, exactly as agents/detector_agent.py's
analyse() does — against every row of the untouched holdout, and reporting
ground-truth precision/recall/F1 on whatever those two layers can resolve
without an LLM call.

What this deliberately does NOT do: call the LLM for the ~85,443-row holdout's
borderline cases. That's neither cheap nor reproducible, and the point of a
ground-truth HARNESS (as opposed to a one-off manual test) is being able to
re-run it after every change for free. Layer 3 gets a separate, honest
treatment below: a small stratified sample, gated behind OPENAI_API_KEY,
reported but not folded into the headline deterministic-layer numbers.

Usage (from the prism/ repo root):
    python domains/bfsi_fraud/eval_harness.py                  # deterministic layers only (free)
    python domains/bfsi_fraud/eval_harness.py --with-llm-sample  # + a small real-LLM sample (costs API calls)

Writes EVAL_RESULTS.md in this domain's folder either way. Requires
domains/bfsi_fraud/data/train.csv and test_holdout.csv — regenerate with
prepare_data.py first (not committed; see that script's docstring).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from domains.bfsi_fraud.agents.detector_agent import FraudDetectorAgent
from domains.bfsi_fraud.tools.ml_scorer import MLScorer

_HERE = Path(__file__).parent
HOLDOUT_PATH = _HERE / "data" / "test_holdout.csv"
TRAIN_PATH = _HERE / "data" / "train.csv"
RESULTS_PATH = _HERE / "EVAL_RESULTS.md"
RESULTS_JSON_PATH = _HERE / "eval_results.json"
LLM_SAMPLE_SIZE = 150
LLM_SAMPLE_FRAUD_FRACTION = 0.3  # oversample fraud — it's ~0.13% of the real base rate,
# a random sample that size would likely contain zero fraud rows otherwise


@dataclass
class LayerResult:
    """One deterministic layer's verdict for every holdout row it resolved.
    `resolved` is a boolean mask over the full holdout — rows this layer
    couldn't call stay False and fall through to the next layer (or, for
    the ones neither layer resolves, to "would need the LLM")."""

    name: str
    resolved: np.ndarray
    predicted_fraud: np.ndarray  # only meaningful where resolved is True


def _apply_rule_layer(df: pd.DataFrame) -> LayerResult:
    resolved = np.zeros(len(df), dtype=bool)
    predicted_fraud = np.zeros(len(df), dtype=bool)
    for i, row in enumerate(df.itertuples(index=False)):
        verdict = FraudDetectorAgent.rule_based_filter({"Amount": row.Amount, "hour": row.hour})
        if verdict is not None:
            resolved[i] = True
            predicted_fraud[i] = True  # every rule branch is a BLOCK — see detector_agent.py
    return LayerResult("Rule-based pre-filter", resolved, predicted_fraud)


def _apply_ml_layer(scores: np.ndarray, already_resolved: np.ndarray) -> LayerResult:
    resolved = (~already_resolved) & ((scores >= MLScorer.FRAUD_THRESHOLD) | (scores <= MLScorer.LEGIT_THRESHOLD))
    predicted_fraud = resolved & (scores >= MLScorer.FRAUD_THRESHOLD)
    return LayerResult("ML fast-path", resolved, predicted_fraud)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_deterministic_eval() -> dict:
    """The free, fast, always-run part — full holdout, rule + ML layers
    only. Returns a dict with per-layer and combined metrics plus
    LLM-escape coverage, ready to be asserted on in tests or rendered
    into EVAL_RESULTS.md."""
    df = pd.read_csv(HOLDOUT_PATH)
    y = df["is_Fraud"].to_numpy()

    scorer = MLScorer(data_path=str(TRAIN_PATH))
    scores = scorer.score_batch(df)

    rule_layer = _apply_rule_layer(df)
    ml_layer = _apply_ml_layer(scores, rule_layer.resolved)

    resolved = rule_layer.resolved | ml_layer.resolved
    predicted_fraud = np.where(rule_layer.resolved, rule_layer.predicted_fraud, ml_layer.predicted_fraud)

    escaped_to_llm = ~resolved

    return {
        "holdout_size": len(df),
        "base_fraud_rate": float(y.mean()),
        "rule_layer": {
            "resolved_count": int(rule_layer.resolved.sum()),
            **(
                _metrics(y[rule_layer.resolved], rule_layer.predicted_fraud[rule_layer.resolved])
                if rule_layer.resolved.any()
                else {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            ),
        },
        "ml_layer": {
            "resolved_count": int(ml_layer.resolved.sum()),
            **(
                _metrics(y[ml_layer.resolved], ml_layer.predicted_fraud[ml_layer.resolved])
                if ml_layer.resolved.any()
                else {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            ),
        },
        "combined_deterministic": {
            "coverage_pct": float(resolved.mean() * 100),
            **_metrics(y[resolved], predicted_fraud[resolved]),
        },
        "escaped_to_llm": {
            "count": int(escaped_to_llm.sum()),
            "pct_of_holdout": float(escaped_to_llm.mean() * 100),
            "fraud_among_escaped": int(y[escaped_to_llm].sum()),
            "fraud_rate_among_escaped": float(y[escaped_to_llm].mean()) if escaped_to_llm.any() else 0.0,
        },
    }


def run_llm_sample_eval(seed: int = 42) -> dict | None:
    """A small, honestly-labeled real-LLM check — NOT part of the free
    regression-tested numbers above. Stratified: oversamples fraud rows so
    a 150-row sample actually contains some (the real base rate is
    ~0.13%, so a uniform random sample this size would very likely
    contain zero). Skipped with a clear message if OPENAI_API_KEY isn't
    set, same "zero-setup default, optional paid extension" split as the
    rest of this codebase's tests."""
    if not os.getenv("OPENAI_API_KEY"):
        print("  [eval_harness] OPENAI_API_KEY not set — skipping the LLM-sample check.")
        return None

    df = pd.read_csv(HOLDOUT_PATH)
    rng = random.Random(seed)

    fraud_df = df[df["is_Fraud"] == 1]
    legit_df = df[df["is_Fraud"] == 0]
    n_fraud = min(len(fraud_df), int(LLM_SAMPLE_SIZE * LLM_SAMPLE_FRAUD_FRACTION))
    n_legit = LLM_SAMPLE_SIZE - n_fraud

    sample = pd.concat(
        [
            fraud_df.sample(n=n_fraud, random_state=seed),
            legit_df.sample(n=n_legit, random_state=seed),
        ]
    ).sample(frac=1, random_state=seed)  # shuffle so layer order isn't correlated with row order

    print(f"  [eval_harness] Running the full 3-layer pipeline on {len(sample)} sampled "
          f"transactions ({n_fraud} fraud, {n_legit} legit)...")

    agent = FraudDetectorAgent(name="EvalHarnessAgent", data_path=str(_HERE / "data" / "transactions_balanced.csv"))
    y_true, y_pred = [], []
    layer_hit = {"rule": 0, "ml": 0, "llm": 0}
    llm_failures = 0
    start = time.perf_counter()

    for row in sample.itertuples(index=False):
        transaction = row._asdict()
        rule_verdict = FraudDetectorAgent.rule_based_filter(transaction)
        if rule_verdict is not None:
            layer_hit["rule"] += 1
            pred = True
        else:
            ml_verdict = agent.ml_filter(transaction)
            if ml_verdict is not None:
                layer_hit["ml"] += 1
                pred = "HIGH" in ml_verdict
            else:
                # Deliberately NOT agent.analyse() here — that method
                # swallows API errors internally and returns a canned
                # "MEDIUM/FLAG" fallback string that's indistinguishable
                # from a real LLM verdict once parsed. Good production
                # behavior (a user-facing system should degrade
                # gracefully), but exactly wrong for an eval harness: a
                # bad API key would silently produce a fake-looking
                # "recall 1.0" from every borderline case defaulting to
                # "flag it" — found for real while building this, see
                # eval_harness.py's module docstring. Call the LLM
                # directly so a real failure is countable, not disguised.
                ml_score = agent.scorer.score(transaction)
                try:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    messages = [
                        SystemMessage(content="You are a bank fraud detection expert. Be concise and precise."),
                        HumanMessage(content=agent.build_prompt(transaction, ml_score=ml_score)),
                    ]
                    raw = agent.llm.invoke(messages).content
                    layer_hit["llm"] += 1
                    parsed = FraudDetectorAgent.parse_response(raw)
                    pred = FraudDetectorAgent.is_fraud(parsed)
                except Exception as e:
                    llm_failures += 1
                    print(f"  [eval_harness] LLM call failed, excluding this row from metrics: {e}")
                    continue  # excluded from y_true/y_pred entirely — not counted either way
        y_true.append(bool(row.is_Fraud))
        y_pred.append(pred)

    elapsed_s = time.perf_counter() - start

    if llm_failures and llm_failures == layer_hit["llm"] + llm_failures:
        # Every single LLM call in this sample failed — there is nothing
        # honest left to report for this section at all.
        print(f"  [eval_harness] All {llm_failures} LLM call(s) in this sample failed "
              "(likely a bad/expired OPENAI_API_KEY) — nothing real to report.")
        return {"sample_size": len(sample), "llm_failures": llm_failures, "all_failed": True}

    return {
        "sample_size": len(sample),
        "llm_failures": llm_failures,
        "all_failed": False,
        "layer_hit_counts": layer_hit,
        "elapsed_seconds": round(elapsed_s, 1),
        **_metrics(np.array(y_true, dtype=int), np.array(y_pred, dtype=int)),
    }


def render_markdown(deterministic: dict, llm_sample: dict | None) -> str:
    d = deterministic
    lines = [
        "# Evaluation Results — Fraud Detection (bfsi_fraud domain)",
        "",
        f"Generated by `eval_harness.py` against `{HOLDOUT_PATH.name}` — a temporal holdout the model "
        "never trained on, at its real, naturally-imbalanced base rate (see `prepare_data.py`).",
        "",
        "## Deterministic layers (rule-based pre-filter + ML fast-path) — free, always run",
        "",
        f"- Holdout size: **{d['holdout_size']:,}** transactions, "
        f"**{d['base_fraud_rate']:.4%}** real fraud rate",
        f"- **{d['combined_deterministic']['coverage_pct']:.1f}%** of transactions resolved without "
        "ever calling the LLM (Layer 3 only runs on the remainder)",
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
        f"Confusion matrix (combined deterministic layers): TP={d['combined_deterministic']['tp']}, "
        f"FP={d['combined_deterministic']['fp']}, FN={d['combined_deterministic']['fn']}, "
        f"TN={d['combined_deterministic']['tn']}",
        "",
        "### What falls through to the LLM",
        "",
        f"- **{d['escaped_to_llm']['count']:,}** transactions ({d['escaped_to_llm']['pct_of_holdout']:.1f}% "
        "of the holdout) are genuinely borderline (ML score between "
        f"{MLScorer.LEGIT_THRESHOLD} and {MLScorer.FRAUD_THRESHOLD}) and never reach a rule match — "
        "these are the ones Layer 3 exists for.",
        f"- Of those, **{d['escaped_to_llm']['fraud_among_escaped']}** are actually fraud "
        f"({d['escaped_to_llm']['fraud_rate_among_escaped']:.2%} of the escaped set — much higher than "
        "the holdout's overall base rate, which is exactly why routing them to a slower, more careful "
        "reasoning step is the right design instead of a single global threshold).",
        "",
    ]

    if llm_sample is not None and llm_sample.get("all_failed"):
        lines += [
            "## Full 3-layer pipeline — stratified sample, real LLM calls",
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
            "## Full 3-layer pipeline — stratified sample, real LLM calls",
            "",
            f"A {llm_sample['sample_size']}-row sample, oversampled for fraud "
            f"({int(LLM_SAMPLE_FRAUD_FRACTION * 100)}% fraud vs. the holdout's real "
            f"{d['base_fraud_rate']:.2%} — a uniform random sample this size would very likely contain "
            "zero fraud rows and tell you nothing). This is the only number in this file that reflects "
            "the LLM's actual borderline judgment, not just the deterministic layers above." + failure_note,
            "",
            f"- Layer hits: rules={lh['rule']}, ML fast-path={lh['ml']}, **LLM={lh['llm']}**",
            f"- Wall time: {llm_sample['elapsed_seconds']}s for {llm_sample['sample_size']} transactions",
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
            "## Full 3-layer pipeline (with real LLM calls)",
            "",
            "Skipped — set `OPENAI_API_KEY` and re-run with `--with-llm-sample` to include a stratified "
            "sample that exercises the actual LLM borderline-case reasoning, not just the two "
            "deterministic layers above.",
            "",
        ]

    lines += [
        "---",
        "*Regenerate: `python domains/bfsi_fraud/eval_harness.py` (add `--with-llm-sample` for the LLM-sample "
        "section). See `evaluate.py` for raw-classifier PR-AUC/ROC-AUC/cost-threshold calibration — a "
        "different, complementary question from what this file answers.*",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm-sample", action="store_true", help="also run a real-LLM stratified sample")
    args = parser.parse_args()

    print(f"Loading holdout: {HOLDOUT_PATH}")
    deterministic = run_deterministic_eval()
    print(f"  Deterministic coverage: {deterministic['combined_deterministic']['coverage_pct']:.1f}%")
    print(f"  Combined precision={deterministic['combined_deterministic']['precision']:.4f} "
          f"recall={deterministic['combined_deterministic']['recall']:.4f} "
          f"f1={deterministic['combined_deterministic']['f1']:.4f}")

    llm_sample = run_llm_sample_eval() if args.with_llm_sample else None

    markdown = render_markdown(deterministic, llm_sample)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"\nWrote {RESULTS_PATH}")

    # Also dump raw numbers as JSON alongside, for anything that wants to
    # consume this programmatically (e.g. a future CI badge) without
    # scraping markdown.
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"deterministic": deterministic, "llm_sample": llm_sample}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
