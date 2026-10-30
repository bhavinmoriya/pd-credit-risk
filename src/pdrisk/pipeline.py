"""End-to-end, reproducible run:  `uv run pd-run`  (add --fast for a ~1 minute smoke run).

Order matters and is deliberate -- the test split is used exactly once, after every modelling decision
(feature handling, hyper-parameters, model choice, cut-off) was fixed on the development (train) split.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import __version__
from .baseline import coefficient_table, fit_baseline, forest_plot, model_summary
from .config import FIG_DIR, MODEL_DIR, REPORT_DIR, SEED, TABLE_DIR, CVConfig, ensure_dirs
from .data import DATA_DICTIONARY, FEATURES, PROTECTED, load_data, split
from .decisions import (GRADE_LABELS, base_rate_sensitivity, benchmark_policies, cost_ratio_sensitivity, decide, empirical_threshold,
                        grade_table, optimal_threshold, plot_cost_curve)
from .eda import run_eda
from .evaluate import (bootstrap_ci, calibration, discrimination, metric_table, paired_bootstrap_diff, plot_discrimination, plot_reliability)
from .explain import (attributions, global_importance, permutation_table, plot_dependence, plot_importance_comparison, plot_local, reason_codes)
from .fairness import audit, oof_mean_prediction, plot_group_rates, proxy_test, unawareness_experiment
from .limitations import learning_curve, plot_learning_curve, selection_bias_experiment
from .models import SPEC_BY_NAME, SPECS, corrected_resampled_ttest, fit_candidates, nested_cv, predict_pd, select_model

warnings.filterwarnings("ignore", category=UserWarning)


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / f"{name}.csv")
    return df


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_bundle(chosen: str, X, y, cfg: CVConfig, thr: float, test_metrics: dict) -> dict:
    """Fit the chosen specification on ALL labelled data and package everything the app needs."""
    spec = SPEC_BY_NAME[chosen]
    keep = [c for c in FEATURES if c not in PROTECTED]
    bundle = {"selected": chosen, "label": spec.label, "threshold": thr, "sample_default_rate": float(y.mean()),
              "protected": PROTECTED, "version": __version__, "test_metrics": test_metrics,
              "background": X.sample(200, random_state=SEED), "data_dictionary": DATA_DICTIONARY,
              "numeric_ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.select_dtypes("number")},
              "categories": {c: sorted(X[c].unique()) for c in X.select_dtypes("object")}}
    for tag, feats in (("full", None), ("no_protected", keep)):
        m = fit_candidates(X, y, cfg, names=[chosen], features=feats)[chosen]
        # explain the raw score of the base learner (the monotone Platt step does not change which features drive it)
        base = fit_candidates(X, y, cfg, names=[spec.base], features=feats)[spec.base] if spec.calibrate else m
        bundle[f"model_{tag}"], bundle[f"explain_model_{tag}"] = m, base
    return bundle


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the full PD modelling pipeline.")
    ap.add_argument("--fast", action="store_true", help="smaller CV / fewer bootstraps (smoke test)")
    args = ap.parse_args(argv)
    cfg = CVConfig.fast() if args.fast else CVConfig()
    ensure_dirs()
    t0 = time.time()
    R: dict = {"seed": SEED, "fast": args.fast}

    # ------------------------------------------------------------------ 1. data
    log("1/9 data")
    X, y = load_data()
    Xtr, Xte, ytr, yte = split(X, y)
    pd.DataFrame({"row": X.index, "set": np.where(X.index.isin(Xtr.index), "train", "test")}).to_csv(TABLE_DIR / "split.csv", index=False)
    R["n_train"], R["n_test"] = len(Xtr), len(Xte)
    R["default_rate_train"], R["default_rate_test"] = float(ytr.mean()), float(yte.mean())
    log(f"   train {len(Xtr)} ({ytr.sum()} defaults) | test {len(Xte)} ({yte.sum()} defaults)")

    # ------------------------------------------------------------------ 2. EDA (train only)
    log("2/9 exploratory statistics (training split only)")
    run_eda(Xtr, ytr)

    # ------------------------------------------------------------------ 3. interpretable baseline
    log("3/9 interpretable baseline (unpenalised logit, statsmodels)")
    base = fit_baseline(Xtr, ytr)
    coef = save(coefficient_table(base), "baseline_coefficients")
    summ = save(model_summary(base).to_frame("value"), "baseline_summary")
    forest_plot(coef, FIG_DIR / "baseline_forest.png")
    R["baseline_n_parameters"] = int(summ.loc["n_parameters", "value"])
    R["events_per_variable"] = float(ytr.sum() / (R["baseline_n_parameters"] - 1))

    # ------------------------------------------------------------------ 4. nested CV model comparison
    log(f"4/9 nested CV on development set ({cfg.outer_splits}x{cfg.outer_repeats} outer folds, {cfg.inner_splits}-fold inner tuning)")
    fm, oof, hp = nested_cv(Xtr, ytr, cfg)
    save(fm, "cv_fold_metrics"); save(hp, "cv_fold_hyperparameters")
    k = fm["fold"].nunique()
    summary = fm.groupby("model")[["auc", "log_loss", "brier"]].agg(["mean", lambda s: s.std(ddof=1) / np.sqrt(len(s))])
    summary.columns = [f"{a}_{'mean' if b == '<lambda_0>' or b == 'mean' else 'se'}" if b == "mean" else f"{a}_se" for a, b in summary.columns]
    summary = summary.loc[[s.name for s in SPECS]]
    save(summary, "cv_summary")
    # discrimination and calibration of pooled OOF predictions, averaged over repeats
    d_rows, c_rows = {}, {}
    for s in SPECS:
        ds = [discrimination(ytr.values, oof[s.name][r]) for r in range(cfg.outer_repeats)]
        cs = [calibration(ytr.values, oof[s.name][r]) for r in range(cfg.outer_repeats)]
        d_rows[s.name] = pd.DataFrame(ds).mean(); c_rows[s.name] = pd.DataFrame(cs).mean()
    save(pd.DataFrame(d_rows).T, "cv_oof_discrimination"); save(pd.DataFrame(c_rows).T, "cv_oof_calibration")
    # corrected paired tests vs the logistic-regression baseline
    wide = {m: fm[fm.model == m].set_index("fold") for m in fm.model.unique()}
    tt = []
    for s in SPECS:
        if s.name == "lr":
            continue
        row = {"model": s.name}
        for met in ("auc", "log_loss"):
            d = (wide[s.name][met] - wide["lr"][met]).values
            t, p = corrected_resampled_ttest(d, int(wide["lr"]["n_train"].iloc[0]), int(wide["lr"]["n_test"].iloc[0]))
            row.update({f"{met}_diff_vs_lr": d.mean(), f"{met}_p_corrected": p})
        tt.append(row)
    save(pd.DataFrame(tt).set_index("model"), "cv_vs_lr_corrected_ttest")
    chosen, sel = select_model(fm)
    save(sel, "model_selection")
    R["selected_model"] = chosen
    log(f"   selected by pre-declared 1-SE rule: {chosen} ({SPEC_BY_NAME[chosen].label})")

    # ------------------------------------------------------------------ 5. one-shot test evaluation
    log("5/9 hold-out test evaluation (single look)")
    models = fit_candidates(Xtr, ytr, cfg)
    pred = {n: predict_pd(m, Xte) for n, m in models.items()}
    pred["logit_unpenalised"] = base.predict(Xte)
    pd.DataFrame({"y": yte.values, **pred}, index=Xte.index).to_csv(TABLE_DIR / "test_predictions.csv")
    disc, cal = metric_table(yte.values, pred)
    save(disc, "test_discrimination"); save(cal, "test_calibration")
    ci_rows = {}
    for n, p in pred.items():
        ci_rows[n] = {}
        for met, fn in (("auc", roc_auc_score), ("log_loss", lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6))), ("brier", brier_score_loss)):
            lo, hi = bootstrap_ci(yte.values, p, fn, cfg.bootstrap, seed=SEED)
            ci_rows[n].update({f"{met}_ci_low": lo, f"{met}_ci_high": hi})
    save(pd.DataFrame(ci_rows).T, "test_bootstrap_ci")
    pdiff = {}
    for n in pred:
        if n == "lr":
            continue
        pdiff[n] = {}
        for met, fn in (("auc", roc_auc_score), ("log_loss", lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6)))):
            m, lo, hi = paired_bootstrap_diff(yte.values, pred[n], pred["lr"], fn, cfg.bootstrap, seed=SEED)
            pdiff[n].update({f"{met}_diff_vs_lr": m, f"{met}_lo": lo, f"{met}_hi": hi})
    save(pd.DataFrame(pdiff).T, "test_paired_diff_vs_lr")
    show = list(dict.fromkeys(["lr", "lr_spline", "rf", "lgbm", chosen]))
    plot_discrimination(yte.values, {n: pred[n] for n in show}, FIG_DIR / "test_discrimination.png", "Discrimination on the hold-out set")
    plot_reliability(yte.values, {n: pred[n] for n in ["lr", "lr_spline", "rf", "rf_cal", "lgbm", "lgbm_cal", "lr_balanced"]},
                     FIG_DIR / "test_reliability.png", "Calibration (reliability) on the hold-out set -- deciles of predicted PD")
    pc = pred[chosen]
    R["test"] = {"auc": float(disc.loc[chosen, "auc"]), "gini": float(disc.loc[chosen, "gini"]), "ks": float(disc.loc[chosen, "ks"]),
                 "brier": float(cal.loc[chosen, "brier"]), "log_loss": float(cal.loc[chosen, "log_loss"]), "ece": float(cal.loc[chosen, "ece"]),
                 "cal_slope": float(cal.loc[chosen, "cal_slope"]), "citl": float(cal.loc[chosen, "citl"]),
                 "auc_ci": [ci_rows[chosen]["auc_ci_low"], ci_rows[chosen]["auc_ci_high"]]}

    # ------------------------------------------------------------------ 6. decisions
    log("6/9 decisions")
    thr = optimal_threshold()
    p_dev = oof[chosen].mean(axis=0)
    thr_emp = empirical_threshold(ytr.values, p_dev)
    plot_cost_curve(ytr.values, p_dev, FIG_DIR / "decision_cost_curve.png", thr, thr_emp)
    pol = save(benchmark_policies(yte.values, pc, thr), "decision_policies_test")
    save(grade_table(yte.values, pc), "decision_grades_test")
    save(cost_ratio_sensitivity(yte.values, pc), "decision_cost_ratio_sensitivity")
    save(base_rate_sensitivity(yte.values, pc, float(ytr.mean())), "decision_base_rate_sensitivity")
    band = pd.DataFrame({"band": decide(pc, thr, 0.05), "y": yte.values}).groupby("band").agg(n=("y", "size"), observed_default=("y", "mean"))
    save(band, "decision_three_way_test")
    R["decision"] = {"analytic_threshold": thr, "empirical_threshold_dev_oof": thr_emp}

    # ------------------------------------------------------------------ 7. explainability
    log("7/9 explainability")
    a_lr, _, _ = attributions(models["lr"], Xte, Xtr)
    a_gb, _, _ = attributions(models["lgbm"], Xte, Xtr)
    imp = {"Logistic regression": global_importance(a_lr), "LightGBM (SHAP)": global_importance(a_gb)}
    plot_importance_comparison(imp, FIG_DIR / "explain_importance.png")
    plot_dependence({"Logistic regression": a_lr, "LightGBM (SHAP)": a_gb}, Xte, FIG_DIR / "explain_dependence.png")
    both = pd.concat(imp, axis=1).dropna()
    rho = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])
    R["importance_rank_corr_lr_vs_lgbm"] = float(rho.statistic)
    save(pd.concat(imp, axis=1), "explain_global_importance")
    save(permutation_table(models[chosen], Xte, yte, n_repeats=15 if args.fast else 30), "explain_permutation_importance_test")
    a_ch, b_ch, units = attributions(models[chosen], Xte, Xtr)
    order = np.argsort(pc)
    examples = {}
    for tag, i in (("lowest_pd", order[0]), ("highest_pd", order[-1]), ("near_cutoff", int(np.argmin(np.abs(pc - thr))))):
        row, x = a_ch.iloc[i], Xte.iloc[i]
        plot_local(row, x, float(pc[i]), FIG_DIR / f"explain_local_{tag}.png")
        examples[tag] = {"row_id": int(Xte.index[i]), "pd": float(pc[i]), "actual_default": int(yte.iloc[i]),
                         "decision": str(decide([pc[i]], thr)[0]), "units": units, "reason_codes": reason_codes(row, x)}
    (TABLE_DIR / "explain_local_examples.json").write_text(json.dumps(examples, indent=2))

    # ------------------------------------------------------------------ 8. fairness
    log("8/9 fairness audit (out-of-fold predictions on all 1000 applicants)")
    Xall, yall = X, y
    p_all = oof_mean_prediction(models[chosen], Xall, yall, n_repeats=1 if args.fast else 3)
    gm, disp = audit(yall.values, p_all, Xall, thr, cfg.bootstrap)
    for attr, t in gm.items():
        save(t, f"fairness_group_metrics_{attr}")
    save(disp.set_index(["attribute", "group", "metric"]), "fairness_disparities")
    plot_group_rates(gm, FIG_DIR / "fairness_groups.png")
    save(unawareness_experiment(models["lr"], Xall, yall, n_repeats=1 if args.fast else 3), "fairness_unawareness")
    save(proxy_test(Xall, n_repeats=1 if args.fast else 3), "fairness_proxy_test")

    # ------------------------------------------------------------------ 9. limitations + deployment bundle
    log("9/9 limitations experiments and deployment bundle")
    save(selection_bias_experiment(Xtr, ytr, Xte, yte, cfg).set_index(["model", "training sample", "evaluated on"]), "limits_selection_bias")
    lc = learning_curve(Xtr, ytr, cfg, repeats=3 if args.fast else 6)
    save(lc.groupby(["model", "n_train"])["auc"].agg(["mean", "std"]), "limits_learning_curve")
    plot_learning_curve(lc, FIG_DIR / "limits_learning_curve.png")
    bundle = build_bundle(chosen, X, y, cfg, thr, R["test"])
    joblib.dump(bundle, MODEL_DIR / "pd_model.joblib", compress=3)

    R["runtime_seconds"] = round(time.time() - t0, 1)
    (REPORT_DIR / "results.json").write_text(json.dumps(R, indent=2, default=float))
    write_report(R, chosen)
    log(f"done in {R['runtime_seconds']}s -> {REPORT_DIR}/REPORT.md, models/pd_model.joblib")


def _t(name: str, digits: int = 3, index: bool = True) -> str:
    return md_table(pd.read_csv(TABLE_DIR / f"{name}.csv", index_col=0 if index else None), digits, index)


def write_report(R: dict, chosen: str) -> None:
    """Auto-generated results appendix: every number comes straight from the run that produced it."""
    parts = [f"# Auto-generated results (seed {R['seed']}, pdrisk {__version__})\n",
             f"Selected model (pre-declared 1-SE rule on nested-CV log loss): **{chosen}** - {SPEC_BY_NAME[chosen].label}.\n",
             "## Data quality (training split)\n" + _t("eda_data_quality"),
             "## Information value\n" + _t("eda_information_value"),
             "## Baseline logit summary\n" + _t("baseline_summary"),
             "## Nested CV (development set): mean and SE over outer folds\n" + _t("cv_summary", 4),
             "### Pooled out-of-fold discrimination\n" + _t("cv_oof_discrimination"),
             "### Pooled out-of-fold calibration\n" + _t("cv_oof_calibration"),
             "### Differences vs logistic regression (Nadeau-Bengio corrected t-test)\n" + _t("cv_vs_lr_corrected_ttest", 4),
             "## Hold-out test: discrimination\n" + _t("test_discrimination"),
             "## Hold-out test: calibration\n" + _t("test_calibration"),
             "### Bootstrap 95% CIs\n" + _t("test_bootstrap_ci", 3),
             "### Paired bootstrap differences vs LR\n" + _t("test_paired_diff_vs_lr", 4),
             "## Decisions: policies on the test set\n" + _t("decision_policies_test"),
             "### Risk grades\n" + _t("decision_grades_test"),
             "### Cost-ratio sensitivity\n" + _t("decision_cost_ratio_sensitivity"),
             "### Portfolio base-rate sensitivity\n" + _t("decision_base_rate_sensitivity"),
             "### Three-way (approve / refer / decline, +-5pp band)\n" + _t("decision_three_way_test"),
             "## Fairness: unawareness\n" + _t("fairness_unawareness"), "### Proxy test\n" + _t("fairness_proxy_test"),
             "### Disparities (bootstrap 95% CI)\n" + _t("fairness_disparities", 3),
             "## Limitations: selection-bias simulation\n" + _t("limits_selection_bias"),
             "## Figures\n" + "\n".join(f"![{p.stem}](figures/{p.name})" for p in sorted(FIG_DIR.glob("*.png")))]
    (REPORT_DIR / "REPORT.md").write_text("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
