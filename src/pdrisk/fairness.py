"""Fairness audit. Protected / proxy attributes in this data: age, sex (inside 'personal_status_sex') and
foreign_worker. This is a statistical audit, not a legal assessment (EU/German anti-discrimination law,
e.g. AGG and Directive 2004/113/EC on sex, needs legal review).

Metrics are computed on out-of-fold predictions so no applicant is scored by a model that saw them.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from .config import SEED
from .data import PROTECTED, sex_from_status

AGE_BINS = [0, 25, 35, 50, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50+"]


def group_frame(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"sex": sex_from_status(X), "age_group": pd.cut(X["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str),
                         "foreign_worker": X["foreign_worker"]}, index=X.index)


def oof_mean_prediction(model, X, y, n_repeats: int = 3) -> np.ndarray:
    """Average of out-of-fold PDs over repeated CV (each row always scored by models that never saw it)."""
    preds = []
    for r in range(n_repeats):
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        preds.append(cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1])
    return np.mean(preds, axis=0)


def _rates(y, p, thr):
    approve = p < thr
    good, bad = y == 0, y == 1
    return {"n": len(y), "observed_default": y.mean(), "mean_pd": p.mean(), "calibration_gap(pred-obs)": p.mean() - y.mean(),
            "approval_rate": approve.mean(),
            "false_reject_rate(good declined)": (~approve & good).sum() / max(good.sum(), 1),
            "default_catch_rate(bad declined)": (~approve & bad).sum() / max(bad.sum(), 1),
            "auc": roc_auc_score(y, p) if good.sum() >= 5 and bad.sum() >= 5 else np.nan}


LEVEL_ORDER = {"sex": ["male", "female"], "age_group": AGE_LABELS, "foreign_worker": ["yes", "no"]}


def _levels(group: pd.Series) -> list[str]:
    present = set(pd.unique(np.asarray(group)))
    order = LEVEL_ORDER.get(group.name, [])
    return [l for l in order if l in present] + sorted(present - set(order))


def group_metrics(y, p, group: pd.Series, thr: float) -> pd.DataFrame:
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(group)
    return pd.DataFrame({lev: _rates(y[g == lev], p[g == lev], thr) for lev in _levels(group)}).T


def disparity_table(y, p, group: pd.Series, thr: float, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Each group vs the largest group (reference): approval-rate ratio (disparate impact, 0.8 'four-fifths'
    heuristic), difference in false-reject rate among creditworthy applicants (equal opportunity for the
    creditworthy), and difference in calibration gap. Bootstrap 95% CIs -- wide, because groups are small."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(group)
    levels = _levels(group)
    ref = max(levels, key=lambda l: (g == l).sum())
    rng = np.random.default_rng(seed)

    def stats_(idx):
        out = {}
        for lev in levels:
            if lev == ref:
                continue
            a, b = idx[g[idx] == lev], idx[g[idx] == ref]
            ra, rb = _rates(y[a], p[a], thr), _rates(y[b], p[b], thr)
            out[lev] = (ra["approval_rate"] / rb["approval_rate"],
                        ra["false_reject_rate(good declined)"] - rb["false_reject_rate(good declined)"],
                        ra["calibration_gap(pred-obs)"] - rb["calibration_gap(pred-obs)"])
        return out

    point = stats_(np.arange(len(y)))
    boots = [stats_(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    rows = []
    for lev, vals in point.items():
        b = np.array([bt[lev] for bt in boots if lev in bt])
        for k, name in enumerate(["approval_rate_ratio", "false_reject_rate_diff", "calibration_gap_diff"]):
            lo, hi = np.nanpercentile(b[:, k], [2.5, 97.5])
            rows.append({"group": lev, "reference": ref, "metric": name, "value": vals[k], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def audit(y, p, X: pd.DataFrame, thr: float, n_boot: int = 1000):
    gf = group_frame(X)
    metrics, disp = {}, []
    for col in gf.columns:
        metrics[col] = group_metrics(y, p, gf[col], thr)
        d = disparity_table(y, p, gf[col], thr, n_boot)
        d.insert(0, "attribute", col)
        disp.append(d)
    return metrics, pd.concat(disp, ignore_index=True)


def unawareness_experiment(model, X, y, n_repeats: int = 3) -> pd.DataFrame:
    """Does dropping the protected attributes cost accuracy, and are they recoverable from the rest (proxies)?"""
    keep = [c for c in X.columns if c not in PROTECTED]
    rows = []
    for name, cols in (("all features", list(X.columns)), ("without protected attributes", keep)):
        m = clone(model)
        m = _restrict_features(m, cols)
        p = oof_mean_prediction(m, X, y, n_repeats)
        rows.append({"model_inputs": name, "oof_auc": roc_auc_score(y, p), "oof_log_loss": log_loss(y, p),
                     "approval_rate@1/6": float((p < 1 / 6).mean())})
    return pd.DataFrame(rows).set_index("model_inputs")


def _restrict_features(model, cols):
    """Rebuild a fitted/unfitted lr-type pipeline on a feature subset, keeping its tuned hyper-parameters."""
    from .config import CVConfig
    from .models import base_estimators
    params = {k: v for k, v in model.get_params().items() if k == "clf__C"}
    est = base_estimators(CVConfig(), cols)["lr"][0]
    return est.set_params(**params)


def proxy_test(X: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    """Cross-validated AUC of predicting each protected attribute from the *non-protected* features.
    AUC well above 0.5 => the remaining features act as proxies; dropping the attribute is then not enough."""
    from sklearn.linear_model import LogisticRegression
    from .config import CVConfig
    from .models import make_preprocessor
    from sklearn.pipeline import Pipeline
    keep = [c for c in X.columns if c not in PROTECTED]
    gf = group_frame(X)
    targets = {"female": (gf["sex"] == "female"), "age<25": (gf["age_group"] == "<25"),
               "age 50+": (gf["age_group"] == "50+"), "non-foreign worker": (gf["foreign_worker"] == "no")}
    pipe = Pipeline([("prep", make_preprocessor("linear", keep)), ("clf", LogisticRegression(C=0.1, max_iter=3000))])
    rows = {}
    for name, t in targets.items():
        aucs = []
        for r in range(n_repeats):
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
            aucs.append(roc_auc_score(t, cross_val_predict(clone(pipe), X, t.astype(int), cv=cv, method="predict_proba")[:, 1]))
        rows[name] = {"prevalence": float(t.mean()), "proxy_auc": float(np.mean(aucs))}
    return pd.DataFrame(rows).T


def plot_group_rates(metrics: dict[str, pd.DataFrame], path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.8))
    for ax, (attr, m) in zip(np.atleast_1d(axes), metrics.items()):
        x = np.arange(len(m)); w = 0.27
        ax.bar(x - w, m["observed_default"], w, label="observed default", color="#C44E52")
        ax.bar(x, m["mean_pd"], w, label="mean predicted PD", color="#4C72B0")
        ax.bar(x + w, 1 - m["approval_rate"], w, label="decline rate", color="#8172B2")
        ax.set_xticks(x); ax.set_xticklabels([f"{i}\n(n={int(n)})" for i, n in zip(m.index, m["n"])], fontsize=7)
        ax.set_title(attr, fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
