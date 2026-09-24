# DriftDetector for payroll -- see domains/payroll_hr/payroll/tools/drift_detector.py.
# Deliberately NOT wired into PayrollDetectorAgent's verdicts: measured
# legit-outlier rate on tier_ml_test_holdout.csv is 10.4% (vs Banking's
# 4.5% bar, see domains/banking/fraud/agents/detector_agent.py's
# analyse()) -- more than double Banking's noise floor, and the closest
# of the three unwired domains, but still not the "rare, meaningful
# signal" Banking's own measured rate earned. See
# test_insurance_drift_detector.py's module docstring for how all of
# these figures were measured the same way.

import pandas as pd
import pytest

from domains.payroll_hr.payroll.tools.drift_detector import DriftDetector

HOLDOUT = "domains/payroll_hr/payroll/data/tier_ml_test_holdout.csv"


@pytest.fixture(scope="module")
def detector():
    d = DriftDetector()
    if not d.trained:
        pytest.skip("training data not generated (run domains/payroll_hr/payroll/data/prepare_data.py)")
    return d


def test_trained_without_label(detector):
    assert "Is_Flagged" not in detector.feature_cols


def test_record_and_batch_agree(detector):
    df = pd.read_csv(HOLDOUT).head(50)
    batch = detector.score_batch(df) < 0
    single = [detector.is_outlier(r) for r in df.to_dict(orient="records")]
    assert list(batch) == single


def test_extreme_record_is_outlier(detector):
    row = pd.read_csv(HOLDOUT).iloc[0].to_dict()
    for col in detector.feature_cols:
        row[col] = 1e9
    assert detector.is_outlier(row)


def test_untrained_detector_never_flags():
    d = DriftDetector(data_path="does_not_exist.csv")
    assert not d.trained and d.is_outlier({}) is False


def test_measured_legit_outlier_rate_is_too_noisy_to_wire_in(detector):
    """Pins the 10.4% figure the module docstring above cites -- see
    test_insurance_drift_detector.py's equivalent test for why this is a
    living check rather than a one-off measurement."""
    holdout = pd.read_csv(HOLDOUT)
    legit = holdout[holdout["Is_Flagged"] == 0]
    rate = (detector.score_batch(legit) < 0).mean()
    assert rate > 0.05
