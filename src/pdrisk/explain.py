"""Explaining model predictions on the 20 *business* features (one-hot columns are summed back).

* Linear models: exact attributions  coef_j * (z_j - mean_j)  in log-odds space (== SHAP for independent features).
* Tree models:   SHAP TreeExplainer (exact for trees; path-dependent feature perturbation).
* Model-agnostic: permutation importance on the raw inputs (drop in AUC / rise in log loss when a feature is shuffled).

Caveat (also in the report): attributions describe how the MODEL uses features, not causal effects on
default, and with correlated features credit can be shared arbitrarily between them.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from .config import SEED


def _prep_and_clf(model):
    """Return (prep_pipeline, classifier) for a fitted pipeline (unwrapping calibration if present)."""
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    return model[:-1], model[-1]


def feature_groups(prep) -> list[str]:
    """Raw feature name for every column of the transformed matrix (ColumnTransformer column order)."""
    ct: ColumnTransformer = prep.named_steps["prep"].named_steps["prep"]
    groups: list[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder":
            continue
        cols = list(cols)
        if name == "cat":
            for feat, cats in zip(cols, trans.categories_):
                groups += [feat] * len(cats)
        elif name == "spl":
            k = trans.named_steps["s"].n_features_out_ // len(cols)
            for feat in cols:
                groups += [feat] * k
        else:
            groups += cols
    return groups


def _aggregate(values: np.ndarray, groups: list[str], order: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(values).T.groupby(np.array(groups)).sum().T
    return df[order]


def attributions(model, X: pd.DataFrame, X_background: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    """Per-applicant, per-feature contributions plus base value and units.

    Units are log-odds for linear models and GBMs; random forests are explained in probability units
    (that is how shap decomposes sklearn forests), so the units string must be carried along.
    """
    prep, clf = _prep_and_clf(model)
    z, zb = prep.transform(X), prep.transform(X_background)
    groups = feature_groups(prep)
    order = list(dict.fromkeys(groups))
    units = "log-odds"
    if isinstance(clf, LogisticRegression):
        w = clf.coef_.ravel()
        contrib = (z.values - zb.values.mean(0)) * w
        base = float(clf.intercept_[0] + zb.values.mean(0) @ w)
    else:
        import shap
        ex = shap.TreeExplainer(clf)
        sv = ex.shap_values(z)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        ev = np.atleast_1d(ex.expected_value)
        contrib, base = sv, float(ev[-1])
        if clf.__class__.__name__ == "RandomForestClassifier":
            units = "probability"
    return _aggregate(contrib, groups, order).set_index(X.index), base, units


def global_importance(attr: pd.DataFrame) -> pd.Series:
    return attr.abs().mean().sort_values(ascending=False)


def permutation_table(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 30) -> pd.DataFrame:
    out = {}
    for scoring, name in (("roc_auc", "auc_drop"), ("neg_log_loss", "logloss_increase")):
        r = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=SEED, n_jobs=1)
        out[name] = pd.Series(r.importances_mean, index=X.columns)
        out[name + "_sd"] = pd.Series(r.importances_std, index=X.columns)
    return pd.DataFrame(out).sort_values("auc_drop", ascending=False)


def reason_codes(contrib_row: pd.Series, x_row: pd.Series, top: int = 4) -> list[str]:
    """Adverse-action style reasons: the features pushing this applicant's PD up the most."""
    pos = contrib_row[contrib_row > 0].sort_values(ascending=False).head(top)
    return [f"{f} = {x_row[f]}  (+{v:.2f} log-odds)" for f, v in pos.items()]


def plot_importance_comparison(imp: dict[str, pd.Series], path, top: int = 12) -> None:
    feats = pd.concat(imp, axis=1).fillna(0)
    feats = feats.loc[feats.mean(axis=1).sort_values().index].tail(top)
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.8 / len(imp)
    for i, c in enumerate(feats.columns):
        ax.barh(np.arange(len(feats)) + i * h, feats[c], height=h, label=c)
    ax.set_yticks(np.arange(len(feats)) + h * (len(imp) - 1) / 2); ax.set_yticklabels(feats.index)
    ax.set_xlabel("mean |contribution| to log-odds (test set)"); ax.legend(); ax.set_title("Global importance by model")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_dependence(attrs: dict[str, pd.DataFrame], X: pd.DataFrame, path, feats=("duration", "amount", "age")) -> None:
    fig, axes = plt.subplots(1, len(feats), figsize=(4.3 * len(feats), 3.6))
    for ax, f in zip(axes, feats):
        for name, a in attrs.items():
            ax.scatter(X.loc[a.index, f], a[f], s=9, alpha=0.5, label=name)
        ax.axhline(0, color="k", lw=0.7); ax.set_xlabel(f); ax.set_ylabel("contribution (log-odds)")
        if f == "amount":
            ax.set_xscale("log")
    axes[0].legend(fontsize=7); fig.suptitle("How each model uses the continuous features (test set)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_local(contrib_row: pd.Series, x_row: pd.Series, pd_hat: float, path=None, top: int = 8):
    s = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([f"{f} = {x_row[f]}" for f in s.index], s.values, color=np.where(s.values > 0, "#C44E52", "#4C72B0"))
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("contribution to log-odds of default")
    ax.set_title(f"Predicted PD = {pd_hat:.1%}", fontsize=10); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130); plt.close(fig)
    return fig
