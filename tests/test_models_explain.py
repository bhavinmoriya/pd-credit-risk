import numpy as np
import pytest

from pdrisk.config import CVConfig
from pdrisk.explain import attributions, feature_groups
from pdrisk.models import fit_candidates, predict_pd


@pytest.fixture(scope="module")
def fitted(splits):
    Xtr, Xte, ytr, yte = splits
    return fit_candidates(Xtr, ytr, CVConfig.fast(), names=["lr", "lr_spline", "lgbm"]), Xtr, Xte


def test_probabilities_are_valid(fitted):
    ms, _, Xte = fitted
    for m in ms.values():
        p = predict_pd(m, Xte)
        assert p.shape == (len(Xte),) and np.all((p > 0) & (p < 1))


def test_feature_groups_cover_transformed_columns(fitted):
    ms, Xtr, _ = fitted
    for m in ms.values():
        assert len(feature_groups(m[:-1])) == m[:-1].transform(Xtr.head(5)).shape[1]


@pytest.mark.parametrize("name", ["lr", "lr_spline", "lgbm"])
def test_attributions_are_additive(fitted, name):
    """base value + sum of feature contributions must reproduce the model's log-odds exactly."""
    ms, Xtr, Xte = fitted
    attr, base, units = attributions(ms[name], Xte.head(40), Xtr)
    p = predict_pd(ms[name], Xte.head(40))
    assert units == "log-odds" and attr.shape[1] == 20
    assert np.allclose(attr.sum(axis=1) + base, np.log(p / (1 - p)), atol=1e-4)


def test_models_use_no_target_information(fitted):
    """Only application-time columns go in; 'default' / 'class' must never be a model input."""
    ms, Xtr, _ = fitted
    assert "default" not in Xtr.columns and "class" not in Xtr.columns
