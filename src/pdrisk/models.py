"""Pre-processing, candidate models, hyper-parameter tuning and nested cross-validation.

Design choices
--------------
* One shared raw-feature interface: every model is an sklearn Pipeline that takes the *raw decoded
  DataFrame*, so explanations / permutation importance operate on the 20 business features.
* Tuning criterion is log loss (a proper scoring rule: rewards good ranking AND good calibration).
* Nested CV: hyper-parameters are tuned in an inner loop that never sees the outer test fold, so the
  outer estimate is not optimistically biased by tuning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from .config import SEED, CVConfig
from .data import CATEGORICAL, NUMERIC


def _log_amount(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["amount"] = np.log(X["amount"])      # loan amount is right-skewed (EDA): model log(DM)
    return X


def _clean_name(n: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", n)).strip("_")


def _clean_columns(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM forbids characters such as < : / , in feature names; keep names readable but safe."""
    X = X.copy()
    X.columns = [_clean_name(c) for c in X.columns]
    if X.columns.duplicated().any():
        raise ValueError("Sanitised feature names collide")
    return X


def _clean_names_out(transformer, input_features):
    return [_clean_name(c) for c in input_features]


def make_preprocessor(kind: str, features: list[str] | None = None) -> Pipeline:
    feats = features or NUMERIC + CATEGORICAL
    num = [c for c in NUMERIC if c in feats]
    cat = [c for c in CATEGORICAL if c in feats]
    smooth = [c for c in ("duration", "amount", "age") if c in num]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "linear":
        ct = ColumnTransformer([("num", StandardScaler(), num), ("cat", ohe, cat)])
    elif kind == "spline":      # GAM-like: smooth non-linear effects for the three continuous drivers
        rest = [c for c in num if c not in smooth]
        ct = ColumnTransformer([
            ("spl", Pipeline([("s", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
                              ("z", StandardScaler())]), smooth),
            ("num", StandardScaler(), rest), ("cat", ohe, cat)])
    elif kind == "tree":
        ct = ColumnTransformer([("num", "passthrough", num), ("cat", ohe, cat)])
    else:
        raise ValueError(kind)
    ct.set_output(transform="pandas")
    return Pipeline([("log", FunctionTransformer(_log_amount, feature_names_out="one-to-one")), ("prep", ct),
                     ("clean", FunctionTransformer(_clean_columns, feature_names_out=_clean_names_out))])


def _pipe(kind: str, clf, features=None) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(kind, features)), ("clf", clf)])


def base_estimators(cfg: CVConfig, features=None) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyper-parameter grid). Grids are deliberately small: n=750 training rows."""
    lr = LogisticRegression(max_iter=3000)
    lr_grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
    rf = RandomForestClassifier(n_estimators=cfg.rf_trees, random_state=SEED, n_jobs=1)
    lgbm = LGBMClassifier(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                          random_state=SEED, verbose=-1, n_jobs=1)
    return {
        "lr": (_pipe("linear", lr, features), lr_grid),
        "lr_spline": (_pipe("spline", lr, features), lr_grid),
        "rf": (_pipe("tree", rf, features), {"clf__min_samples_leaf": [3, 8, 15],
                                              "clf__max_features": ["sqrt", 0.4]}),
        "lgbm": (_pipe("tree", lgbm, features), {"clf__n_estimators": [80, 200], "clf__num_leaves": [3, 7],
                                                  "clf__min_child_samples": [15, 30]}),
    }


@dataclass(frozen=True)
class Spec:
    name: str
    label: str
    base: str
    complexity: int                 # used by the one-standard-error rule: lower = simpler
    calibrate: bool = False
    balanced: bool = False
    selectable: bool = True


SPECS = [
    Spec("lr", "Logistic regression (L2)", "lr", 0),
    Spec("lr_spline", "Logistic regression + splines", "lr_spline", 1),
    Spec("rf", "Random forest", "rf", 2),
    Spec("rf_cal", "Random forest + Platt", "rf", 2, calibrate=True),
    Spec("lgbm", "LightGBM", "lgbm", 3),
    Spec("lgbm_cal", "LightGBM + Platt", "lgbm", 3, calibrate=True),
    # Deliberately mis-specified: class re-weighting keeps ranking but destroys PD calibration.
    Spec("lr_balanced", "LR class_weight='balanced' (demo)", "lr", 0, balanced=True, selectable=False),
]
SPEC_BY_NAME = {s.name: s for s in SPECS}


def fit_candidates(X: pd.DataFrame, y: pd.Series, cfg: CVConfig, names: list[str] | None = None,
                   features=None) -> dict[str, object]:
    """Tune each base model once by inner CV (log loss), then derive calibrated variants."""
    bases = base_estimators(cfg, features)
    inner = StratifiedKFold(cfg.inner_splits, shuffle=True, random_state=SEED)
    wanted = [s for s in SPECS if names is None or s.name in names]
    tuned: dict[str, GridSearchCV] = {}
    for b in {s.base for s in wanted}:
        est, grid = bases[b]
        tuned[b] = GridSearchCV(est, grid, scoring="neg_log_loss", cv=inner, n_jobs=1, refit=True).fit(X, y)
    out = {}
    for s in wanted:
        gs = tuned[s.base]
        if s.calibrate:
            m = CalibratedClassifierCV(clone(gs.best_estimator_), method="sigmoid", cv=5).fit(X, y)
        elif s.balanced:
            m = clone(gs.best_estimator_).set_params(clf__class_weight="balanced").fit(X, y)
        else:
            m = gs.best_estimator_
        m.best_params_ = gs.best_params_
        out[s.name] = m
    return out


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def nested_cv(X: pd.DataFrame, y: pd.Series, cfg: CVConfig, verbose: bool = True):
    """Repeated stratified outer CV with inner tuning. Returns fold metrics and out-of-fold PDs.

    oof[name] has shape (n_repeats, n): each repeat gives every row exactly one out-of-fold prediction.
    """
    outer = RepeatedStratifiedKFold(n_splits=cfg.outer_splits, n_repeats=cfg.outer_repeats, random_state=SEED)
    names = [s.name for s in SPECS]
    oof = {n: np.full((cfg.outer_repeats, len(X)), np.nan) for n in names}
    rows, fold_hparams = [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        rep = k // cfg.outer_splits
        models = fit_candidates(X.iloc[tr], y.iloc[tr], cfg)
        for n, m in models.items():
            p = predict_pd(m, X.iloc[te])
            oof[n][rep, te] = p
            rows.append({"fold": k, "repeat": rep, "model": n, "n_train": len(tr), "n_test": len(te),
                         "auc": roc_auc_score(y.iloc[te], p), "log_loss": log_loss(y.iloc[te], p),
                         "brier": brier_score_loss(y.iloc[te], p)})
            fold_hparams.append({"fold": k, "model": n, **m.best_params_})
        if verbose:
            print(f"  outer fold {k + 1}/{outer.get_n_splits()} done", flush=True)
    return pd.DataFrame(rows), oof, pd.DataFrame(fold_hparams)


def select_model(fold_metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Pre-declared rule: simplest model whose mean CV log loss is within 1 SE of the best."""
    ll = fold_metrics.pivot(index="fold", columns="model", values="log_loss")
    cands = [s for s in SPECS if s.selectable]
    mean = ll[[s.name for s in cands]].mean()
    best = mean.idxmin()
    se_best = ll[best].std(ddof=1) / np.sqrt(len(ll))
    ok = [s for s in cands if mean[s.name] <= mean[best] + se_best]
    chosen = sorted(ok, key=lambda s: (s.complexity, mean[s.name]))[0].name
    table = pd.DataFrame({"mean_log_loss": mean, "within_1se_of_best": [n in {s.name for s in ok}
                                                                          for n in mean.index]})
    table["complexity"] = [SPEC_BY_NAME[n].complexity for n in table.index]
    return chosen, table


def corrected_resampled_ttest(d: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected t-test for paired CV differences d (accounts for overlap)."""
    from scipy import stats
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = np.mean(d) / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))
