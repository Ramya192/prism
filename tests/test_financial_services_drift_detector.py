# DriftDetector for financial_services -- see tools/drift_detector.py.
# Deliberately NOT wired into FintechDetectorAgent's verdicts: measured
# legit-outlier rates (13.9% Tier 1, 12.0% Tier 2, vs Banking's 4.5% bar)
# make it noise, and Tier 2's outlier rate is identical for fraud and legit.

import pandas as pd
import pytest

from domains.financial_services.tools.drift_detector import DriftDetector

HOLDOUTS = {
    "tier1": "domains/financial_services/data/test_holdout.csv",
    "tier2": "domains/financial_services/data/tier2_test_holdout.csv",
}


@pytest.fixture(scope="module", params=["tier1", "tier2"])
def detector(request):
    d = DriftDetector(request.param)
    if not d.trained:
        pytest.skip("training data not generated (run prepare_data.py / prepare_tier2_data.py)")
    return d


def test_trained_without_label(detector):
    assert "is_scam" not in detector.feature_cols and "is_manipulative" not in detector.feature_cols


def test_record_and_batch_agree(detector):
    df = pd.read_csv(HOLDOUTS[detector.tier]).head(50)
    batch = detector.score_batch(df) < 0
    single = [detector.is_outlier(r) for r in df.to_dict(orient="records")]
    assert list(batch) == single


def test_extreme_record_is_outlier(detector):
    row = pd.read_csv(HOLDOUTS[detector.tier]).iloc[0].to_dict()
    for col in detector.feature_cols:
        if col not in detector.scorer_cls.CATEGORICAL_COLS:
            row[col] = 1e9
    assert detector.is_outlier(row)


def test_untrained_detector_never_flags():
    d = DriftDetector("tier1", data_path="does_not_exist.csv")
    assert not d.trained and d.is_outlier({}) is False
