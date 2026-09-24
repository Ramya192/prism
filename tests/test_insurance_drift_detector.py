# DriftDetector for insurance -- see domains/insurance/tools/drift_detector.py.
# Deliberately NOT wired into HealthcareDetectorAgent's verdicts: measured
# legit-outlier rate on test_holdout.csv is 51.1% (vs Banking's 4.5% bar,
# see domains/banking/fraud/agents/detector_agent.py's analyse()) -- the
# caveat would fire on roughly half of legitimate claims, pure noise
# rather than a meaningful signal. The 4.5%/13.9%/12.0%/10.4%/43.2%/51.1%
# figures across Banking/FinServ-T1/FinServ-T2/Payroll/HR/Insurance are
# all measured the same way: is_outlier() rate over each domain's own
# holdout's legitimate (non-fraud) rows.

import pandas as pd
import pytest

from domains.insurance.tools.claim_scorer import ClaimFraudScorer
from domains.insurance.tools.drift_detector import DriftDetector

HOLDOUT = "domains/insurance/data/test_holdout.csv"


@pytest.fixture(scope="module")
def detector():
    d = DriftDetector()
    if not d.trained:
        pytest.skip("training data not generated (run domains/insurance/data/prepare_data.py)")
    return d


def test_trained_without_label(detector):
    assert "Is_Fraud" not in detector.feature_cols


def test_record_and_batch_agree(detector):
    df = pd.read_csv(HOLDOUT).head(50)
    batch = detector.score_batch(df) < 0
    single = [detector.is_outlier(r) for r in df.to_dict(orient="records")]
    assert list(batch) == single


def test_extreme_record_is_outlier(detector):
    row = pd.read_csv(HOLDOUT).iloc[0].to_dict()
    for col in detector.feature_cols:
        if col not in ClaimFraudScorer.CATEGORICAL_COLS:
            row[col] = 1e9
    assert detector.is_outlier(row)


def test_untrained_detector_never_flags():
    d = DriftDetector(data_path="does_not_exist.csv")
    assert not d.trained and d.is_outlier({}) is False


def test_measured_legit_outlier_rate_is_too_noisy_to_wire_in(detector):
    """Pins the 51.1% figure the module docstring above cites. If a future
    retrain drops this materially (well below ~15%, Banking-order-of-
    magnitude), the "deliberately not wired in" decision should be
    revisited rather than left stale -- this test is what would fail to
    prompt that."""
    holdout = pd.read_csv(HOLDOUT)
    legit = holdout[holdout["Is_Fraud"] == 0]
    rate = (detector.score_batch(legit) < 0).mean()
    assert rate > 0.3
