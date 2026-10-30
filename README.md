# pdrisk: probability of default (PD) on the Statlog German Credit data

A reproducible, tested Python project that goes from raw credit data to a documented PD model, decisions,
explanations, a fairness audit and a list of what the model can *not* be trusted to do.
Managed with **uv**; everything derives from `SEED = 42`.

```bash
uv sync                    # install (Python >= 3.10)
uv run pd-run              # full pipeline, ~4 min on 1 CPU  (uv run pd-run --fast: ~1 min smoke run)
uv run pytest              # 23 tests
uv run pd-app              # Gradio app (launches with share=True)
```
Outputs: `reports/REPORT.md` (every table, auto-generated), `reports/tables/*.csv`, `reports/figures/*.png`,
`models/pd_model.joblib`. Data documentation: [`DATA_CARD.md`](DATA_CARD.md).

## Layout
| module | step |
|---|---|
| `data.py` | download + SHA-256 verification, decoding from the original UCI codes, target definition, split |
| `eda.py` | default rates with Wilson CIs, chi-square/Cramér's V, Mann-Whitney, information value, VIF (training split only) |
| `baseline.py` | unpenalised logit (statsmodels): odds ratios, CIs, LR test |
| `models.py` | candidates, nested CV, pre-declared selection rule, Nadeau-Bengio corrected tests |
| `evaluate.py` | discrimination (AUC/Gini/KS) and calibration (Brier, log loss, ECE, slope, CITL, Hosmer-Lemeshow) *separately* |
| `decisions.py` | cost-based cut-off, risk grades, approve/refer/decline, prior-shift (base-rate) adjustment |
| `explain.py` | exact linear attributions, SHAP for trees, permutation importance, reason codes |
| `fairness.py` | group metrics, bootstrap disparities, unawareness and proxy tests |
| `limitations.py` | selection-bias simulation, learning curve |
| `app.py` | Gradio scorer |

## Method in brief
* **Target:** `default = 1` iff UCI class 2 ("bad"). Polarity is unit-tested. The data has no default definition, horizon or dates, so no out-of-time validation is possible (see data card).
* **Protocol:** stratified 75/25 split. All choices (hyper-parameters, model, cut-off) are made on the 750-row development set with *nested* repeated CV (5x2 outer, 3-fold inner, tuned on log loss). The 250-row test set is used once for the final report. (During development I ran one smoke test that printed test AUCs; no choice depended on it, and the selection rule is fixed in code.)
* **Candidates:** L2 logistic regression; LR + splines; random forest; LightGBM; Platt-calibrated RF / LightGBM; plus a deliberately mis-specified LR with `class_weight="balanced"` as a calibration demonstration (not selectable).
* **Selection rule (pre-declared):** simplest model whose mean CV log loss is within 1 SE of the best.

## Findings (seed 42; regenerate with `uv run pd-run`)
**Data.** Provenance matters: two public decoded copies of this dataset are wrong (mislabelled `status`/`purpose`; scrambled `personal_status_sex`). We decode the original UCI codes ourselves and test marginals against UCI counts.

**Statistics.** Strongest signals: checking-account status (information value 0.62), credit history, savings, duration. Baseline logit odds ratio for `status < 0 DM` vs "no checking account" is 5.2 (CI 3.0-8.8). With 49 parameters on 225 defaults (4.7 events per variable) an unpenalised logit is over-parameterised; the L2 version is preferable.

**Model comparison.** Nested-CV AUC is 0.773 (LR) to 0.784 (calibrated RF); *no* model differs significantly from LR (corrected p >= 0.38 for AUC and log loss). On the hold-out set LR gets AUC **0.80** (95% CI 0.73-0.86), Gini 0.60, KS 0.50, Brier 0.157; paired bootstrap differences of the alternatives vs LR all include zero. So the selected model is **L2 logistic regression**. Caveat: the learning curve is still rising at ~560 rows and LightGBM catches up at the largest size, so "LR is enough" is a statement about *this* sample size.

**Discrimination vs calibration.** `class_weight="balanced"` leaves AUC unchanged (0.804 vs 0.801) but wrecks calibration: mean predicted PD 0.45 vs 0.30 observed, calibration-in-the-large -0.83, Hosmer-Lemeshow p ~ 0, log loss +0.059 (CI excludes 0). Selected LR is well calibrated (CITL -0.02, slope 1.18; ECE 0.075 is noisy at n=250). Platt scaling did not materially help any model.

**Decisions.** With the UCI cost matrix (false accept : false reject = 5 : 1), approve iff PD < 1/6. This analytic cut-off matches the empirical optimum on development out-of-fold predictions (17%). On the test set: expected cost 0.57 per applicant vs 0.91 for a naive 50% cut-off, 1.50 for approving everyone and 0.70 for rejecting everyone; 34% approved, 10.5% default among approved, 88% of defaults caught, 98 good applicants rejected. The advantage over "reject all" is modest because the 30% sample default rate makes lending look expensive. Risk grades are monotone in observed default (6% -> 66%) though grade C (n=36) is over-predicted. **Decisions are more sensitive to the assumed portfolio default rate than to model choice:** re-basing PDs to a 5% portfolio rate raises approvals at the same cut-off from 34% to 91%.

**Explanations.** LR and LightGBM-SHAP agree on drivers (Spearman 0.87 on global importance): status, savings, duration, credit history. SHAP shows what LR cannot: a U-shaped effect of amount and a rise in risk after age 50. Per-applicant reason codes are in the app and `reports/tables/explain_local_examples.json`. Attributions describe the model, not causes.

**Fairness (statistical audit, not legal advice).** Predictions are calibrated within sex, age and foreign-worker groups (calibration gaps' CIs include 0), but outcomes differ: women are approved at 0.71x the rate of men (CI 0.56-0.87), under-25s at 0.63x vs 25-34 (CI 0.41-0.89), and creditworthy women/under-25s are rejected ~10-12 pp more often (CIs exclude 0). Because base rates differ, calibration and equal error rates cannot both hold (Kleinberg et al.; Chouldechova). Dropping age, sex-bearing status and foreign-worker costs little (OOF AUC 0.786 -> 0.781), but these attributes are recoverable from the remaining features (proxy AUC 0.70-0.80), so removal is not a fix by itself. The app ships a model variant without protected inputs. Groups are small (37 non-foreign workers; 4 defaults), so CIs are wide.

**Limitations.** (1) Selection bias: training on applicants a stricter legacy scorecard would have accepted drops test AUC 0.80 -> 0.77 and under-predicts PD by 17 pp (0.37 vs 0.54) in the region it never saw, for LR and LightGBM alike. Every row here is an *accepted* loan, so the whole dataset carries this problem. (2) Non-representative base rate (30%). (3) 1970s data, DM amounts, undocumented default definition and horizon. (4) n = 1,000: wide CIs, little power to separate models. (5) Single bank, no time dimension, so no stability or drift monitoring. (6) Not usable for real lending decisions.

## Reproducibility notes
Raw data is committed and checksum-pinned; tuning, splits, bootstraps and CV all use `SEED`. Results can differ in the last digits across library versions (versions are pinned by `uv.lock`).
