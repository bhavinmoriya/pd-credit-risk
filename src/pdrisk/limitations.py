"""Experiments that probe what the model can NOT be trusted to do."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from .config import SEED, CVConfig
from .evaluate import calibration_slope_intercept
from .models import _pipe, predict_pd


def _fixed_models(cfg: CVConfig):
    return {"LR (C=0.1)": _pipe("linear", LogisticRegression(C=0.1, max_iter=3000)),
            "LightGBM (fixed)": _pipe("tree", LGBMClassifier(n_estimators=80, num_leaves=7, min_child_samples=30, learning_rate=0.05,
                                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                                              random_state=SEED, verbose=-1, n_jobs=1))}


def selection_bias_experiment(Xtr, ytr, Xte, yte, cfg: CVConfig, reject_share: float = 0.30) -> pd.DataFrame:
    """Simulate reject-inference blindness.

    Real banks only observe outcomes for applicants they accepted. Here every row is an accepted loan, so we can
    *simulate* a stricter historic policy: a crude 3-variable 'legacy scorecard' refuses the riskiest 30% of the
    training applicants, and we retrain on the survivors only. We then score the whole test set and look at the
    region the legacy policy would have refused -- where the retrained model never saw data.
    """
    legacy_cols = ["status", "credit_history", "savings"]
    legacy = _pipe("linear", LogisticRegression(C=1.0, max_iter=3000), features=legacy_cols).fit(Xtr, ytr)
    cut = np.quantile(predict_pd(legacy, Xtr), 1 - reject_share)
    keep = predict_pd(legacy, Xtr) < cut
    region = predict_pd(legacy, Xte) >= cut                       # applicants the legacy policy would refuse
    rows = []
    for name, est in _fixed_models(cfg).items():
        for train_label, idx in (("trained on all applicants", np.ones(len(Xtr), bool)),
                                 ("trained on legacy-accepted only", keep)):
            m = __import__("sklearn.base", fromlist=["clone"]).clone(est).fit(Xtr[idx], ytr[idx])
            p = predict_pd(m, Xte)
            for reg_label, mask in (("whole test set", np.ones(len(Xte), bool)), ("legacy-refused region", region)):
                yy, pp = yte.values[mask], p[mask]
                rows.append({"model": name, "training sample": train_label, "evaluated on": reg_label, "n_train": int(idx.sum()),
                             "n_eval": int(mask.sum()), "auc": roc_auc_score(yy, pp), "mean_pd": pp.mean(), "observed_default": yy.mean(),
                             "pd_bias(pred-obs)": pp.mean() - yy.mean()})
    return pd.DataFrame(rows)


def learning_curve(X, y, cfg: CVConfig, fracs=(0.15, 0.3, 0.5, 0.75, 1.0), repeats: int = 6) -> pd.DataFrame:
    """Validation AUC vs training-set size (within the development set only)."""
    from sklearn.base import clone
    rows = []
    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.25, random_state=SEED)
    for r, (tr, va) in enumerate(sss.split(X, y)):
        rng = np.random.default_rng(SEED + r)
        for f in fracs:
            sub = rng.choice(tr, int(len(tr) * f), replace=False)
            for name, est in _fixed_models(cfg).items():
                m = clone(est).fit(X.iloc[sub], y.iloc[sub])
                rows.append({"frac": f, "n_train": len(sub), "model": name, "auc": roc_auc_score(y.iloc[va], predict_pd(m, X.iloc[va]))})
    return pd.DataFrame(rows)


def plot_learning_curve(lc: pd.DataFrame, path) -> None:
    g = lc.groupby(["model", "n_train"])["auc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, d in g.groupby("model"):
        ax.errorbar(d["n_train"], d["mean"], yerr=d["std"], marker="o", capsize=3, label=name)
    ax.set(xlabel="training rows", ylabel="validation AUC (mean ± sd over splits)", title="Learning curve (development set only)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
