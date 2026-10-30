import pytest

from pdrisk.app import NUMERIC_DEFAULTS, score_applicant
from pdrisk.config import SEED, CVConfig
from pdrisk.data import CATEGORICAL, FEATURES, PROTECTED
from pdrisk.models import fit_candidates


@pytest.fixture(scope="module")
def bundle(data):
    X, y = data
    keep = [c for c in FEATURES if c not in PROTECTED]
    full = fit_candidates(X, y, CVConfig.fast(), names=["lr"])["lr"]
    np_ = fit_candidates(X, y, CVConfig.fast(), names=["lr"], features=keep)["lr"]
    return {"model_full": full, "explain_model_full": full, "model_no_protected": np_, "explain_model_no_protected": np_,
            "sample_default_rate": float(y.mean()), "background": X.sample(100, random_state=SEED),
            "numeric_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in NUMERIC_DEFAULTS}}


def _applicant(bundle):
    a = dict(NUMERIC_DEFAULTS)
    a.update(bundle["background"][CATEGORICAL].mode().iloc[0].to_dict())
    return a


def test_score_returns_decision_and_reasons(bundle):
    r = score_applicant(bundle, _applicant(bundle))
    assert 0 < r["pd"] < 1 and r["decision"] in {"approve", "refer", "decline"} and r["threshold"] == pytest.approx(1 / 6)


def test_riskier_applicant_gets_higher_pd(bundle):
    a = _applicant(bundle); b = dict(a, duration=60, status="< 0 DM", savings="< 100 DM")
    assert score_applicant(bundle, b)["pd"] > score_applicant(bundle, a)["pd"]


def test_prior_shift_lowers_pd_and_range_warning(bundle):
    a = _applicant(bundle)
    assert score_applicant(bundle, a, portfolio_rate=0.05)["pd"] < score_applicant(bundle, a)["pd"]
    assert score_applicant(bundle, dict(a, age=120))["warnings"]


def test_no_protected_variant_ignores_protected_inputs(bundle):
    a = _applicant(bundle)
    b = dict(a, age=22, personal_status_sex="female: divorced/separated/married", foreign_worker="no")
    r1 = score_applicant(bundle, a, "no_protected"); r2 = score_applicant(bundle, b, "no_protected")
    assert r1["pd"] == pytest.approx(r2["pd"])
