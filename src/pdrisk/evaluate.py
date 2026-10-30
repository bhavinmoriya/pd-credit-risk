"""Discrimination and calibration are evaluated SEPARATELY.

Discrimination = ranking ability (does a defaulter get a higher PD than a good customer?):
    AUC / Gini / KS. Invariant to any monotone transformation of the scores.
Calibration = are the numbers themselves probabilities? (do 20% PDs default ~20% of the time?):
    Brier, log loss, ECE, calibration intercept & slope, Hosmer-Lemeshow. NOT invariant.
A model can be excellent on one and poor on the other (see the 'balanced' demo in the report).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- discrimination
def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def discrimination(y, p) -> dict[str, float]:
    auc = roc_auc_score(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y, p)}


# ---------------------------------------------------------------- calibration
def calibration_bins(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-frequency bins (equal-width bins are nearly empty in the tails for PD data)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
    return g.reset_index()


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    g = calibration_bins(y, p, n_bins)
    return float(np.sum(g["n"] * np.abs(g["mean_pred"] - g["obs_rate"])) / g["n"].sum())


def calibration_slope_intercept(y, p) -> dict[str, float]:
    """Cox recalibration: logit P(y=1) = a + b*logit(p).  Ideal: b = 1.
    'citl' (calibration-in-the-large) is the intercept with slope fixed at 1 (offset model); ideal 0.
    b < 1 -> over-confident (too extreme) PDs;  b > 1 -> under-confident."""
    lp = logit(np.asarray(p))
    y = np.asarray(y)
    slope_fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    citl_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return {"cal_slope": float(slope_fit.params[1]), "cal_intercept_given_slope": float(slope_fit.params[0]),
            "citl": float(citl_fit.params[0])}


def hosmer_lemeshow(y, p, g: int = 10) -> tuple[float, float]:
    """HL chi-square on out-of-sample predictions (df = g). Low power at n~250: read with the plots."""
    b = calibration_bins(y, p, g)
    obs, exp, n = b["obs_rate"] * b["n"], b["mean_pred"] * b["n"], b["n"]
    stat = float(np.sum((obs - exp) ** 2 / (exp * (1 - exp / n) + 1e-12)))
    return stat, float(stats.chi2.sf(stat, g))


def calibration(y, p) -> dict[str, float]:
    hl, hl_p = hosmer_lemeshow(y, p)
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
            "ece": expected_calibration_error(y, p), "mean_pred": float(np.mean(p)),
            "obs_rate": float(np.mean(y)), **calibration_slope_intercept(y, p), "hl_stat": hl, "hl_p": hl_p}


def metric_table(y, preds: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.DataFrame({n: discrimination(y, p) for n, p in preds.items()}).T
    c = pd.DataFrame({n: calibration(y, p) for n, p in preds.items()}).T
    return d, c


def bootstrap_ci(y, p, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for any metric fn(y, p) -> float."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(y, p_a, p_b, fn, n_boot: int = 1000, seed: int = 0):
    """CI for fn(a) - fn(b) on the same resamples (paired: far tighter than comparing two CIs)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(fn(y[i], p_a[i]) - fn(y[i], p_b[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- plots
def plot_discrimination(y, preds: dict[str, np.ndarray], path, title="") -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    y = np.asarray(y)
    for n, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(y, p):.3f})")
        order = np.argsort(-p)                         # CAP: share of defaults captured vs share screened
        cap = np.cumsum(y[order]) / y.sum()
        ax[1].plot(np.arange(1, len(y) + 1) / len(y), cap, label=n)
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set(xlabel="Share of applicants (riskiest first)",
                                               ylabel="Share of defaults captured", title="Cumulative accuracy profile")
    ax[0].legend(fontsize=7); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_reliability(y, preds: dict[str, np.ndarray], path, title="", n_bins: int = 10, ncols: int = 4) -> None:
    n = len(preds)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.5 * nrows), sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, (name, p) in zip(flat, preds.items()):
        b = calibration_bins(y, p, n_bins)
        ax.plot([0, 1], [0, 1], "k:")
        ax.plot(b["mean_pred"], b["obs_rate"], "o-", color="#C44E52")
        ax.set_title(f"{name}\nECE={expected_calibration_error(y, p, n_bins):.3f}", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed default rate")
    for ax in axes[-1]:
        ax.set_xlabel("mean predicted PD")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
