import numpy as np
import pytest

from pdrisk.decisions import decide, optimal_threshold, policy_outcomes, prior_shift
from pdrisk.evaluate import calibration_slope_intercept, discrimination, expected_calibration_error, logit


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    p = rng.beta(2, 5, 20000)
    y = rng.binomial(1, p)
    return y, p


def test_perfectly_calibrated_scores(synthetic):
    y, p = synthetic
    c = calibration_slope_intercept(y, p)
    assert abs(c["cal_slope"] - 1) < 0.08 and abs(c["citl"]) < 0.06
    assert expected_calibration_error(y, p) < 0.02


def test_overconfident_scores_have_same_auc_but_worse_calibration(synthetic):
    """The core reason discrimination and calibration are reported separately."""
    y, p = synthetic
    p_over = 1 / (1 + np.exp(-2.5 * logit(p)))          # monotone transform: ranking unchanged
    assert discrimination(y, p)["auc"] == pytest.approx(discrimination(y, p_over)["auc"], abs=1e-12)
    assert calibration_slope_intercept(y, p_over)["cal_slope"] < 0.6
    assert expected_calibration_error(y, p_over) > 3 * expected_calibration_error(y, p)


def test_threshold_uses_cost_ratio():
    assert optimal_threshold(5, 1) == pytest.approx(1 / 6)
    assert optimal_threshold(1, 1) == 0.5


def test_cost_optimal_threshold_minimises_cost_for_calibrated_pds(synthetic):
    y, p = synthetic
    grid = np.linspace(0.02, 0.9, 89)
    costs = [policy_outcomes(y, p, t)["expected_cost_per_applicant"] for t in grid]
    assert abs(grid[int(np.argmin(costs))] - 1 / 6) < 0.04


def test_prior_shift_identity_and_direction():
    p = np.array([0.1, 0.3, 0.6])
    assert np.allclose(prior_shift(p, 0.3, 0.3), p)
    assert np.all(prior_shift(p, 0.3, 0.05) < p) and np.all(prior_shift(p, 0.3, 0.5) > p)
    assert np.all(np.diff(prior_shift(p, 0.3, 0.05)) > 0)      # ranking preserved


def test_three_way_decision():
    out = decide([0.05, 0.16, 0.30], 1 / 6, 0.05)
    assert list(out) == ["approve", "refer", "decline"]
