# core/model_store.py
# Persisted model artifacts for every startup-trained scorer / drift
# detector (the ML tiers under domains/*/tools/).
#
# Why this exists: each scorer used to re-train from its train CSV every
# time a process started. Those CSVs are large (~700MB in total) and
# deliberately gitignored/dockerignored, so on a fresh `git clone` (e.g.
# the EC2 box) they don't exist -- the scorer logged "not found", stayed
# untrained, returned 0.5 for everything, and every record silently fell
# through to the LLM tier. Retraining on a 1GB t3.micro wouldn't be
# realistic even if the CSVs were shipped.
#
# What it does: load_or_train(scorer, data_path) loads a committed joblib
# artifact from models/ if it is valid, otherwise calls the scorer's own
# _train() and (best-effort) saves the result. An artifact is used only if:
#   - it was built with the same sklearn/lightgbm/xgboost versions (pinned in requirements.txt)
#     (a pickle from another version can load wrongly, not just fail), and
#   - the scorer's train args match, and
#   - the train CSV is absent (production) OR its sha256 still matches the
#     one the artifact was trained on (development: edit the data, and the
#     model retrains and rewrites the artifact on the next start).
# Anything else falls back to training exactly as before, so a scorer
# constructed with a test fixture path never reads or clobbers a real
# artifact (artifacts exist only for data files inside this repo, keyed by
# their repo-relative path, and a hash mismatch retrains).
#
# Regenerate every artifact after changing data or model code with:
#   python -m core.model_store --rebuild
#
# Artifacts are pickles: only ever load ones from this repo's own models/
# directory, never from user input.

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("PRISM_MODELS_DIR") or REPO_ROOT / "models")

_VERSIONED_LIBS = ("sklearn", "lightgbm", "xgboost")


def _lib_versions() -> dict[str, str]:
    versions = {}
    for name in _VERSIONED_LIBS:
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # not installed -> not part of the fingerprint
            continue
    return versions


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_path(obj, data_path) -> Path | None:
    """models/<module path>.<ClassName>__<data file path>.joblib -- the
    module path disambiguates the several classes all called DriftDetector,
    the (repo-relative) data path disambiguates one class trained on
    different files. None if the data file lives outside the repo: a test
    fixture or scratch CSV must never read, or overwrite, a real artifact."""
    try:
        rel = Path(str(data_path)).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    module = type(obj).__module__.removeprefix("domains.")
    data_id = rel.with_suffix("").as_posix().replace("/", "-")
    return MODELS_DIR / f"{module}.{type(obj).__name__}__{data_id}.joblib"


def load_or_train(obj, data_path, *train_args, signature: str = "") -> bool:
    """Populates `obj` (a scorer/detector exposing `_train(data_path,
    *train_args)` and a `trained` flag). Returns True if it came from an
    artifact, False if it was trained (or failed to train) from the CSV.
    `signature` folds any constructor state that changes the trained model
    (beyond train_args) into the validity check."""
    data = Path(str(data_path))
    art = artifact_path(obj, data_path)
    data_hash = _sha256(data) if data.exists() else None
    expected = {"versions": _lib_versions(), "args": repr(train_args), "signature": signature}

    if art is not None and art.exists():
        reason = None
        try:
            payload = joblib.load(art)
            meta, state = payload["meta"], payload["state"]
            if meta["versions"] != expected["versions"]:
                reason = f"built with {meta['versions']}, running {expected['versions']}"
            elif meta["args"] != expected["args"] or meta["signature"] != expected["signature"]:
                reason = "trained with different arguments"
            elif data_hash is not None and meta["data_sha256"] != data_hash:
                reason = f"{data.name} changed since the artifact was built"
        except Exception as e:  # corrupt / unreadable -> retrain rather than crash startup
            reason = f"unreadable ({e})"
        if reason is None:
            vars(obj).update(state)
            print(f"  [{type(obj).__name__}] Loaded {art.name}")
            return True
        logger.warning("Ignoring model artifact %s: %s", art.name, reason)

    obj._train(data_path, *train_args)
    if art is not None and getattr(obj, "trained", False) and data_hash is not None:
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"meta": {**expected, "data_sha256": data_hash}, "state": dict(vars(obj))},
                art, compress=3,
            )
            print(f"  [{type(obj).__name__}] Saved {art.name}")
        except Exception as e:  # a read-only image must not break a working scorer
            logger.warning("Could not save model artifact %s: %s", art.name, e)
    return False


def rebuild_all() -> None:
    """Deletes models/*.joblib and re-trains every scorer/detector the app
    constructs by default, by building each domain pipeline's detectors.
    Needs the full train CSVs present (prepare_data.py first)."""
    for p in MODELS_DIR.glob("*.joblib"):
        p.unlink()

    from domains.banking.fraud.tools.ml_scorer import MLScorer
    from domains.banking.fraud.tools.generalizable_scorer import GeneralizableScorer
    from domains.banking.fraud.tools.drift_detector import DriftDetector as BankingDrift
    from domains.insurance.tools.claim_scorer import ClaimFraudScorer
    from domains.insurance.tools.generalizable_scorer import ClaimGeneralizableScorer
    from domains.insurance.tools.drift_detector import DriftDetector as InsuranceDrift
    from domains.financial_services.tools.scam_scorer import ScamScorer
    from domains.financial_services.tools.wash_trading_scorer import WashTradingScorer
    from domains.financial_services.tools.drift_detector import DriftDetector as FsDrift
    from domains.payroll_hr.payroll.tools.anomaly_scorer import PayrollAnomalyScorer
    from domains.payroll_hr.payroll.tools.generalizable_scorer import PayrollGeneralizableScorer
    from domains.payroll_hr.payroll.tools.drift_detector import DriftDetector as PayrollDrift
    from domains.payroll_hr.hr.tools.posting_scorer import PostingFraudScorer
    from domains.payroll_hr.hr.tools.drift_detector import DriftDetector as HrDrift

    MLScorer()
    BankingDrift()
    GeneralizableScorer()
    for cls in (ClaimFraudScorer, ClaimGeneralizableScorer, ScamScorer, WashTradingScorer,
                PayrollAnomalyScorer, PayrollGeneralizableScorer, PostingFraudScorer):
        cls()
    InsuranceDrift()
    FsDrift(tier="tier1")
    FsDrift(tier="tier2")
    PayrollDrift()
    HrDrift()

    total = 0
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        size = p.stat().st_size
        total += size
        print(f"  {size / 1e6:7.2f} MB  {p.name}")
    print(f"  {total / 1e6:7.2f} MB  total ({len(list(MODELS_DIR.glob('*.joblib')))} artifacts)")


if __name__ == "__main__":
    if "--rebuild" in sys.argv:
        rebuild_all()
    else:
        print(__doc__)
