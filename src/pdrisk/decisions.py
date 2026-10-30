"""From probabilities to decisions.

With calibrated PDs the Bayes-optimal rule for a two-action problem is analytic:
    approve  iff  PD * C_FA  <  (1 - PD) * C_FR   <=>   PD < C_FR / (C_FA + C_FR)
where C_FA = cost of approving an applicant who defaults, C_FR = cost of rejecting one who would repay.
With the UCI cost matrix (5:1) the cut-off is 1/6 = 16.7% -- far below the naive 50%.
This is only valid if the PDs are calibrated *to the portfolio the bank actually faces* (see prior_shift).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import COST_FALSE_ACCEPT, COST_FALSE_REJECT
from .evaluate import logit

GRADE_EDGES = [0.0, 0.10, 0.20, 0.30, 0.50, 1.0001]
GRADE_LABELS = ["A (<10%)", "B (10-20%)", "C (20-30%)", "D (30-50%)", "E (>50%)"]


def optimal_threshold(c_fa: float = COST_FALSE_ACCEPT, c_fr: float = COST_FALSE_REJECT) -> float:
    return c_fr / (c_fa + c_fr)


def prior_shift(p, sample_rate: float, target_rate: float):
    """Re-base PDs from the training default rate to the default rate of the target portfolio.

    Assumes only the class prior differs (P(x|y) unchanged): add logit(target) - logit(sample) to the
    log-odds (Saerens et al. 2002 / King & Zeng 2001). A strong, untestable assumption -- use for
    sensitivity analysis, not as a substitute for a representative sample.
    """
    shift = logit(np.array(target_rate)) - logit(np.array(sample_rate))
    z = logit(np.asarray(p)) + shift
    return 1 / (1 + np.exp(-z))


def decide(pd_hat, threshold: float, refer_margin: float = 0.0) -> np.ndarray:
    """approve below threshold-margin, decline above threshold+margin, refer to an analyst in between."""
    p = np.asarray(pd_hat)
    out = np.where(p < threshold - refer_margin, "approve", np.where(p >= threshold + refer_margin, "decline", "refer"))
    return out


def policy_outcomes(y, pd_hat, threshold, c_fa=COST_FALSE_ACCEPT, c_fr=COST_FALSE_REJECT) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(pd_hat)
    approve = p < threshold
    fa, fr = np.sum(approve & (y == 1)), np.sum(~approve & (y == 0))
    n = len(y)
    return {"threshold": threshold, "approval_rate": approve.mean(),
            "bad_rate_among_approved": float(y[approve].mean()) if approve.any() else np.nan,
            "defaults_caught": float(((~approve) & (y == 1)).sum() / max(y.sum(), 1)),
            "false_accepts": int(fa), "false_rejects": int(fr),
            "expected_cost_per_applicant": (c_fa * fa + c_fr * fr) / n}


def benchmark_policies(y, pd_hat, threshold, **kw) -> pd.DataFrame:
    n = len(y)
    return pd.DataFrame({
        "approve everyone": policy_outcomes(y, np.zeros(n), 0.5, **kw),
        "reject everyone": policy_outcomes(y, np.ones(n), 0.5, **kw),
        "model, naive cut-off 50%": policy_outcomes(y, pd_hat, 0.5, **kw),
        f"model, cost-optimal cut-off {threshold:.1%}": policy_outcomes(y, pd_hat, threshold, **kw),
    }).T


def cost_curve(y, pd_hat, grid=None, **kw) -> pd.DataFrame:
    grid = np.linspace(0.02, 0.9, 89) if grid is None else grid
    return pd.DataFrame([policy_outcomes(y, pd_hat, t, **kw) for t in grid])


def empirical_threshold(y, pd_hat, **kw) -> float:
    c = cost_curve(y, pd_hat, **kw)
    return float(c.loc[c["expected_cost_per_applicant"].idxmin(), "threshold"])


def grade_table(y, pd_hat) -> pd.DataFrame:
    g = pd.cut(np.asarray(pd_hat), GRADE_EDGES, labels=GRADE_LABELS, right=False)
    d = pd.DataFrame({"grade": g, "y": np.asarray(y), "p": np.asarray(pd_hat)})
    t = d.groupby("grade", observed=False).agg(n=("y", "size"), mean_pd=("p", "mean"), observed_default=("y", "mean"))
    t["share"] = t["n"] / t["n"].sum()
    return t


def cost_ratio_sensitivity(y, pd_hat, ratios=(1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    for r in ratios:
        t = optimal_threshold(r, 1.0)
        rows.append({"cost_ratio_FA:FR": f"{r}:1", **policy_outcomes(y, pd_hat, t, c_fa=r, c_fr=1.0)})
    return pd.DataFrame(rows).set_index("cost_ratio_FA:FR")


def base_rate_sensitivity(y, pd_hat, sample_rate, target_rates=(0.03, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """What the *same* cost-optimal rule does if the real portfolio default rate differs from the sample's."""
    t = optimal_threshold()
    rows = []
    for r in target_rates:
        shifted = prior_shift(pd_hat, sample_rate, r)
        rows.append({"assumed_portfolio_default_rate": r, "mean_shifted_pd": shifted.mean(),
                     "approval_rate_at_16.7%_cutoff": float((shifted < t).mean()),
                     "approval_rate_unshifted": float((np.asarray(pd_hat) < t).mean())})
    return pd.DataFrame(rows).set_index("assumed_portfolio_default_rate")


def plot_cost_curve(y, pd_hat, path, analytic_t, empirical_t) -> None:
    c = cost_curve(y, pd_hat)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(c["threshold"], c["expected_cost_per_applicant"], color="#4C72B0")
    ax[0].axvline(analytic_t, color="g", ls="--", label=f"analytic {analytic_t:.3f}")
    ax[0].axvline(empirical_t, color="r", ls=":", label=f"empirical min {empirical_t:.2f}")
    ax[0].axvline(0.5, color="grey", ls="-.", label="naive 0.5")
    ax[0].set(xlabel="approval cut-off on PD", ylabel="expected cost per applicant (FR=1, FA=5)",
              title="Cost vs cut-off"); ax[0].legend(fontsize=8)
    ax[1].plot(c["approval_rate"], c["bad_rate_among_approved"], color="#C44E52")
    ax[1].set(xlabel="approval rate", ylabel="default rate among approved", title="Acceptance trade-off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
