"""Statistical exploration. Run on the TRAINING split only so the test set stays untouched."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from .config import FIG_DIR, TABLE_DIR
from .data import CATEGORICAL, NUMERIC


def data_quality(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = {
        "n_rows": len(X), "n_features": X.shape[1], "missing_cells": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()), "n_defaults": int(y.sum()),
        "default_rate": float(y.mean()),
    }
    return pd.Series(rows, name="value").to_frame()


def numeric_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-feature location by class, Mann-Whitney test and rank-based AUC (effect size)."""
    out = []
    for c in NUMERIC:
        a, b = X.loc[y == 1, c], X.loc[y == 0, c]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out.append({"feature": c, "median_default": a.median(), "median_good": b.median(),
                    "mean_all": X[c].mean(), "sd_all": X[c].std(), "skew": stats.skew(X[c]),
                    "univariate_auc": u.statistic / (len(a) * len(b)), "mannwhitney_p": u.pvalue})
    return pd.DataFrame(out).set_index("feature")


def categorical_summary(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Default rate per level with Wilson CI; chi-square test and Cramer's V per feature."""
    levels, tests = [], []
    for c in CATEGORICAL:
        g = pd.DataFrame({"x": X[c], "y": y}).groupby("x")["y"].agg(["sum", "count"])
        lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
        levels.append(pd.DataFrame({"feature": c, "level": g.index, "n": g["count"].values,
                                    "default_rate": (g["sum"] / g["count"]).values,
                                    "ci_low": lo, "ci_high": hi}))
        ct = pd.crosstab(X[c], y)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        tests.append({"feature": c, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v})
    return pd.concat(levels, ignore_index=True), pd.DataFrame(tests).set_index("feature")


def information_value(X: pd.DataFrame, y: pd.Series, bins: int = 5) -> pd.DataFrame:
    """Weight-of-evidence / information value (standard credit-scoring screening statistic)."""
    rows = []
    for c in X.columns:
        x = X[c]
        if c in NUMERIC and x.nunique() > 6:
            x = pd.qcut(x, bins, duplicates="drop")
        t = pd.crosstab(x, y).astype(float) + 0.5          # +0.5 smoothing avoids log(0)
        dist_bad, dist_good = t[1] / t[1].sum(), t[0] / t[0].sum()
        iv = float(((dist_good - dist_bad) * np.log(dist_good / dist_bad)).sum())
        rows.append({"feature": c, "information_value": iv})
    iv = pd.DataFrame(rows).set_index("feature").sort_values("information_value", ascending=False)
    iv["strength"] = pd.cut(iv["information_value"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                            labels=["useless", "weak", "medium", "strong", "very strong/suspicious"])
    return iv


def multicollinearity(X: pd.DataFrame) -> pd.DataFrame:
    Z = X[NUMERIC].assign(amount=np.log(X["amount"]))
    Z = (Z - Z.mean()) / Z.std()
    Z.insert(0, "const", 1.0)
    return pd.DataFrame({"VIF": [variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])]},
                        index=NUMERIC)


def _rate_plot(ax, X, y, col, bins=None, rot=30):
    x = pd.qcut(X[col], bins, duplicates="drop") if bins else X[col]
    g = pd.DataFrame({"x": x, "y": y}).groupby("x", observed=True)["y"].agg(["sum", "count"])
    lo, hi = proportion_confint(g["sum"], g["count"], method="wilson")
    r = g["sum"] / g["count"]
    ax.bar(range(len(g)), r, color="#4C72B0", yerr=[r - lo, hi - r], capsize=3)
    ax.axhline(y.mean(), color="k", ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i) for i in g.index], rotation=rot, ha="right", fontsize=7)
    ax.set_title(f"{col} (n per bar: {g['count'].min()}-{g['count'].max()})", fontsize=9)
    ax.set_ylabel("default rate")


def make_figures(X: pd.DataFrame, y: pd.Series, iv: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, bins) in zip(axes.ravel(), [("status", None), ("credit_history", None), ("savings", None),
                                              ("duration", 5), ("amount", 5), ("age", 5)]):
        _rate_plot(ax, X, y, col, bins)
    fig.suptitle("Observed default rate by feature (95% Wilson CI; dashed = overall rate)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_default_rates.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    s = iv["information_value"].sort_values()
    ax.barh(s.index, s.values, color="#55A868")
    for v in (0.02, 0.1, 0.3):
        ax.axvline(v, color="grey", ls=":", lw=1)
    ax.set_xlabel("Information value (training split)"); ax.set_title("Univariate predictive strength")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_information_value.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, c in zip(axes, ["duration", "amount", "age"]):
        for lab, col in ((0, "#4C72B0"), (1, "#C44E52")):
            ax.hist(X.loc[y == lab, c], bins=25, alpha=0.55, density=True, color=col,
                    label="default" if lab else "good")
        ax.set_title(c); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_numeric_distributions.png", dpi=130); plt.close(fig)


def run_eda(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    iv = information_value(X, y)
    levels, cat_tests = categorical_summary(X, y)
    tables = {"data_quality": data_quality(X, y), "numeric_summary": numeric_summary(X, y),
              "categorical_levels": levels, "categorical_tests": cat_tests,
              "information_value": iv, "vif": multicollinearity(X)}
    for name, t in tables.items():
        t.to_csv(TABLE_DIR / f"eda_{name}.csv")
    make_figures(X, y, iv)
    return tables
