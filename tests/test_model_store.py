# core/model_store.py -- committed model artifacts for the startup-trained
# scorers/drift detectors. Fully offline and data-free: it needs no train
# CSV, no API key, and (for the artifact checks) no network, so it runs in
# CI on a fresh clone -- which is exactly the environment the artifacts exist for.

import pytest

import core.model_store as model_store
from domains.banking.fraud.tools.drift_detector import DriftDetector as BankingDrift
from domains.banking.fraud.tools.generalizable_scorer import GeneralizableScorer
from domains.banking.fraud.tools.ml_scorer import MLScorer
from domains.financial_services.tools.drift_detector import DriftDetector as FsDrift
from domains.financial_services.tools.scam_scorer import ScamScorer
from domains.financial_services.tools.wash_trading_scorer import WashTradingScorer
from domains.insurance.tools.claim_scorer import ClaimFraudScorer
from domains.insurance.tools.drift_detector import DriftDetector as InsuranceDrift
from domains.insurance.tools.generalizable_scorer import ClaimGeneralizableScorer
from domains.payroll_hr.hr.tools.drift_detector import DriftDetector as HrDrift
from domains.payroll_hr.hr.tools.posting_scorer import PostingFraudScorer
from domains.payroll_hr.payroll.tools.anomaly_scorer import PayrollAnomalyScorer
from domains.payroll_hr.payroll.tools.drift_detector import DriftDetector as PayrollDrift
from domains.payroll_hr.payroll.tools.generalizable_scorer import PayrollGeneralizableScorer

# Every scorer/detector the app builds by default (see model_store.rebuild_all).
DEFAULT_MODELS = [
    (MLScorer, {}),
    (BankingDrift, {}),
    (GeneralizableScorer, {}),
    (ClaimFraudScorer, {}),
    (ClaimGeneralizableScorer, {}),
    (InsuranceDrift, {}),
    (ScamScorer, {}),
    (WashTradingScorer, {}),
    (FsDrift, {"tier": "tier1"}),
    (FsDrift, {"tier": "tier2"}),
    (PayrollAnomalyScorer, {}),
    (PayrollGeneralizableScorer, {}),
    (PayrollDrift, {}),
    (PostingFraudScorer, {}),
    (HrDrift, {}),
]


@pytest.mark.parametrize("cls,kwargs", DEFAULT_MODELS, ids=lambda v: getattr(v, "__name__", None) or "")
def test_default_model_loads_from_committed_artifact(cls, kwargs, monkeypatch):
    """The regression this whole module exists for: on a fresh clone the
    train CSVs are absent, so every default model must come from an
    artifact. _train is booby-trapped so a silent fall-back to (re)training
    fails loudly -- locally that also catches artifacts gone stale against
    regenerated data (rebuild with `python -m core.model_store --rebuild`)."""
    def no_training(self, *a, **k):
        raise AssertionError(f"{cls.__name__} tried to train instead of loading its artifact")

    monkeypatch.setattr(cls, "_train", no_training)
    model = cls(**kwargs)
    assert model.trained


class _Toy:
    """Smallest thing load_or_train accepts: a _train() and a trained flag."""

    trained = False
    train_calls = 0

    def __init__(self, data_path, factor=1):
        self.factor = factor
        model_store.load_or_train(self, data_path, factor)

    def _train(self, data_path, factor):
        type(self).train_calls += 1
        self.value = factor * 10
        self.trained = True


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(model_store, "REPO_ROOT", tmp_path)   # treat tmp_path as "inside the repo"
    _Toy.train_calls = 0
    csv = tmp_path / "toy.csv"
    csv.write_text("a\n1\n")
    return csv


def test_trains_once_then_loads_artifact(store):
    assert _Toy(store).value == 10
    assert _Toy(store).value == 10
    assert _Toy.train_calls == 1


def test_changed_data_retrains(store):
    _Toy(store)
    store.write_text("a\n1\n2\n")
    _Toy(store)
    assert _Toy.train_calls == 2


def test_missing_csv_uses_artifact(store):
    _Toy(store)
    store.unlink()
    assert _Toy(store).value == 10
    assert _Toy.train_calls == 1


def test_different_train_args_do_not_reuse_artifact(store):
    _Toy(store, factor=1)
    assert _Toy(store, factor=2).value == 20
    assert _Toy.train_calls == 2


def test_library_version_mismatch_ignores_artifact(store, monkeypatch):
    _Toy(store)
    monkeypatch.setattr(model_store, "_lib_versions", lambda: {"sklearn": "0.0.0"})
    _Toy(store)
    assert _Toy.train_calls == 2


def test_corrupt_artifact_retrains_instead_of_crashing(store):
    _Toy(store)
    for f in model_store.MODELS_DIR.glob("*.joblib"):
        f.write_bytes(b"not a pickle")
    assert _Toy(store).value == 10
    assert _Toy.train_calls == 2


def test_missing_csv_and_no_artifact_stays_untrained(store):
    store.unlink()
    toy = _Toy(store)
    assert toy.train_calls == 1 and toy.trained    # _train ran (toy ignores the file) ...
    assert not list(model_store.MODELS_DIR.glob("*.joblib"))   # ... but nothing was saved without data


def test_data_outside_repo_never_touches_artifacts(tmp_path, monkeypatch):
    """A scorer built on a fixture/scratch CSV (whatever it's named) must
    neither load a real artifact nor overwrite one."""
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(model_store, "REPO_ROOT", tmp_path / "repo")
    _Toy.train_calls = 0
    csv = tmp_path / "train.csv"
    csv.write_text("a" + chr(10) + "1" + chr(10))
    _Toy(csv)
    _Toy(csv)
    assert _Toy.train_calls == 2
    assert not (tmp_path / "models").exists()
