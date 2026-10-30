"""Interpretable statistical baseline: unpenalised logistic regression (statsmodels).

Gives what a credit-risk validator expects: odds ratios with confidence intervals, Wald p-values,
a likelihood-ratio test against the null model and McFadden R^2. Reference level of every categorical
feature = its most frequent training level (so odds ratios compare against the 'typical' applicant).
Continuous effects are expressed per interpretable unit: +12 months, doubling of amount, +10 years of age.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .data import CATEGORICAL, NUMERIC


@dataclass
class Baseline:
    result: object
    ref_levels: dict[str, str]
    columns: list[str]

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        return _design(X, self.ref_levels, self.columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.result.predict(self.design(X)))


def _design(X: pd.DataFrame, ref_levels: dict[str, str], columns: list[str] | None = None) -> pd.DataFrame:
    D = pd.DataFrame(index=X.index)
    D["duration (+12 months)"] = X["duration"] / 12
    D["amount (x2)"] = np.log2(X["amount"])
    D["age (+10 years)"] = X["age"] / 10
    for c in ("installment_rate", "present_residence", "number_credits", "people_liable"):
        D[c + " (+1)"] = X[c]
    for c in CATEGORICAL:
        for lev in sorted(X[c].unique()):
            if lev != ref_levels[c]:
                D[f"{c}: {lev}"] = (X[c] == lev).astype(float)
    D = sm.add_constant(D, has_constant="add")
    return D if columns is None else D.reindex(columns=columns, fill_value=0.0)


def fit_baseline(X: pd.DataFrame, y: pd.Series) -> Baseline:
    ref = {c: X[c].value_counts().idxmax() for c in CATEGORICAL}
    D = _design(X, ref)
    res = sm.Logit(y, D).fit(method="newton", maxiter=200, disp=0)
    return Baseline(res, ref, list(D.columns))


def coefficient_table(b: Baseline) -> pd.DataFrame:
    r = b.result
    ci = r.conf_int()
    t = pd.DataFrame({"coef": r.params, "odds_ratio": np.exp(r.params), "or_ci_low": np.exp(ci[0]),
                      "or_ci_high": np.exp(ci[1]), "p_value": r.pvalues})
    return t.drop(index="const")


def model_summary(b: Baseline) -> pd.Series:
    r = b.result
    lr = 2 * (r.llf - r.llnull)
    return pd.Series({"n": int(r.nobs), "n_parameters": int(len(r.params)), "log_likelihood": r.llf,
                      "LR_chi2_vs_null": lr, "LR_p_value": stats.chi2.sf(lr, r.df_model),
                      "mcfadden_r2": r.prsquared, "aic": r.aic, "bic": r.bic,
                      "converged": bool(r.mle_retvals["converged"])})


def forest_plot(t: pd.DataFrame, path, top: int = 22) -> None:
    s = t.reindex(t["coef"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(s) + 1.5))
    sig = s["p_value"] < 0.05
    ax.errorbar(s["odds_ratio"], range(len(s)), xerr=[s["odds_ratio"] - s["or_ci_low"], s["or_ci_high"] - s["odds_ratio"]],
                fmt="none", ecolor="grey", capsize=2)
    ax.scatter(s["odds_ratio"], range(len(s)), c=np.where(sig, "#C44E52", "#4C72B0"), zorder=3)
    ax.axvline(1, color="k", lw=1); ax.set_xscale("log")
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
    ax.set_xlabel("odds ratio of default (log scale; red = p<0.05)")
    ax.set_title("Baseline logit: largest effects (reference = most frequent level)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
