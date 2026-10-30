# Auto-generated results (seed 42, pdrisk 0.1.0)


Selected model (pre-declared 1-SE rule on nested-CV log loss): **lr** - Logistic regression (L2).


## Data quality (training split)
| index | value |
|---|---|
| n_rows | 750.000 |
| n_features | 20.000 |
| missing_cells | 0.000 |
| duplicate_rows | 0.000 |
| n_defaults | 225.000 |
| default_rate | 0.300 |

## Information value
| feature | information_value | strength |
|---|---|---|
| status | 0.624 | very strong/suspicious |
| credit_history | 0.260 | medium |
| savings | 0.231 | medium |
| duration | 0.190 | medium |
| property | 0.150 | medium |
| purpose | 0.145 | medium |
| employment_duration | 0.127 | medium |
| other_installment_plans | 0.093 | weak |
| housing | 0.090 | weak |
| amount | 0.070 | weak |
| personal_status_sex | 0.048 | weak |
| foreign_worker | 0.042 | weak |
| age | 0.034 | weak |
| installment_rate | 0.029 | weak |
| job | 0.022 | weak |
| other_debtors | 0.020 | weak |
| number_credits | 0.007 | useless |
| telephone | 0.005 | useless |
| present_residence | 0.001 | useless |
| people_liable | 0.001 | useless |

## Baseline logit summary
| index | value |
|---|---|
| n | 750 |
| n_parameters | 49 |
| log_likelihood | -336.03376032536835 |
| LR_chi2_vs_null | 244.22893246535853 |
| LR_p_value | 4.356613714457346e-28 |
| mcfadden_r2 | 0.2665392097009619 |
| aic | 770.0675206507367 |
| bic | 996.4511077707241 |
| converged | True |

## Nested CV (development set): mean and SE over outer folds
| model | auc_mean | auc_se | log_loss_mean | log_loss_se | brier_mean | brier_se |
|---|---|---|---|---|---|---|
| lr | 0.7731 | 0.0088 | 0.5066 | 0.0068 | 0.1693 | 0.0026 |
| lr_spline | 0.7765 | 0.0078 | 0.5040 | 0.0063 | 0.1683 | 0.0025 |
| rf | 0.7809 | 0.0091 | 0.5059 | 0.0071 | 0.1679 | 0.0026 |
| rf_cal | 0.7844 | 0.0099 | 0.5031 | 0.0075 | 0.1668 | 0.0030 |
| lgbm | 0.7786 | 0.0085 | 0.5049 | 0.0065 | 0.1684 | 0.0028 |
| lgbm_cal | 0.7759 | 0.0088 | 0.5084 | 0.0063 | 0.1698 | 0.0027 |
| lr_balanced | 0.7723 | 0.0092 | 0.5651 | 0.0081 | 0.1930 | 0.0035 |

### Pooled out-of-fold discrimination
| index | auc | gini | ks |
|---|---|---|---|
| lr | 0.773 | 0.546 | 0.435 |
| lr_spline | 0.777 | 0.554 | 0.437 |
| rf | 0.779 | 0.558 | 0.432 |
| rf_cal | 0.782 | 0.564 | 0.443 |
| lgbm | 0.778 | 0.555 | 0.429 |
| lgbm_cal | 0.774 | 0.548 | 0.434 |
| lr_balanced | 0.773 | 0.545 | 0.432 |

### Pooled out-of-fold calibration
| index | brier | log_loss | ece | mean_pred | obs_rate | cal_slope | cal_intercept_given_slope | citl | hl_stat | hl_p |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | 0.169 | 0.507 | 0.026 | 0.301 | 0.300 | 1.031 | 0.015 | -0.006 | 5.201 | 0.843 |
| lr_spline | 0.168 | 0.504 | 0.040 | 0.301 | 0.300 | 1.009 | 0.001 | -0.005 | 11.836 | 0.449 |
| rf | 0.168 | 0.506 | 0.039 | 0.303 | 0.300 | 1.150 | 0.079 | -0.015 | 10.993 | 0.359 |
| rf_cal | 0.167 | 0.503 | 0.028 | 0.303 | 0.300 | 1.064 | 0.025 | -0.017 | 5.352 | 0.847 |
| lgbm | 0.168 | 0.505 | 0.026 | 0.299 | 0.300 | 1.064 | 0.052 | 0.009 | 5.855 | 0.812 |
| lgbm_cal | 0.170 | 0.508 | 0.034 | 0.301 | 0.300 | 1.086 | 0.049 | -0.008 | 7.633 | 0.663 |
| lr_balanced | 0.193 | 0.565 | 0.146 | 0.446 | 0.300 | 0.981 | -0.804 | -0.807 | 92.588 | 0.000 |

### Differences vs logistic regression (Nadeau-Bengio corrected t-test)
| model | auc_diff_vs_lr | auc_p_corrected | log_loss_diff_vs_lr | log_loss_p_corrected |
|---|---|---|---|---|
| lr_spline | 0.0034 | 0.6486 | -0.0027 | 0.7036 |
| rf | 0.0078 | 0.5036 | -0.0007 | 0.9404 |
| rf_cal | 0.0113 | 0.3837 | -0.0036 | 0.6515 |
| lgbm | 0.0054 | 0.5668 | -0.0017 | 0.8171 |
| lgbm_cal | 0.0028 | 0.7648 | 0.0018 | 0.7472 |
| lr_balanced | -0.0008 | 0.6609 | 0.0585 | 0.0000 |

## Hold-out test: discrimination
| index | auc | gini | ks |
|---|---|---|---|
| lr | 0.801 | 0.602 | 0.503 |
| lr_spline | 0.806 | 0.612 | 0.501 |
| rf | 0.780 | 0.559 | 0.465 |
| rf_cal | 0.786 | 0.573 | 0.491 |
| lgbm | 0.790 | 0.580 | 0.432 |
| lgbm_cal | 0.794 | 0.588 | 0.444 |
| lr_balanced | 0.804 | 0.609 | 0.512 |
| logit_unpenalised | 0.790 | 0.580 | 0.497 |

## Hold-out test: calibration
| index | brier | log_loss | ece | mean_pred | obs_rate | cal_slope | cal_intercept_given_slope | citl | hl_stat | hl_p |
|---|---|---|---|---|---|---|---|---|---|---|
| lr | 0.157 | 0.482 | 0.075 | 0.304 | 0.300 | 1.180 | 0.090 | -0.023 | 12.634 | 0.245 |
| lr_spline | 0.156 | 0.479 | 0.051 | 0.302 | 0.300 | 1.106 | 0.053 | -0.012 | 7.368 | 0.690 |
| rf | 0.167 | 0.502 | 0.044 | 0.305 | 0.300 | 1.058 | 0.004 | -0.031 | 4.411 | 0.927 |
| rf_cal | 0.166 | 0.500 | 0.048 | 0.306 | 0.300 | 0.965 | -0.060 | -0.038 | 6.457 | 0.775 |
| lgbm | 0.163 | 0.494 | 0.051 | 0.300 | 0.300 | 1.052 | 0.038 | 0.003 | 6.265 | 0.793 |
| lgbm_cal | 0.162 | 0.490 | 0.048 | 0.299 | 0.300 | 1.106 | 0.075 | 0.005 | 6.551 | 0.767 |
| lr_balanced | 0.182 | 0.541 | 0.150 | 0.450 | 0.300 | 1.154 | -0.858 | -0.830 | 33.682 | 0.000 |
| logit_unpenalised | 0.163 | 0.499 | 0.050 | 0.306 | 0.300 | 0.760 | -0.178 | -0.043 | 12.400 | 0.259 |

### Bootstrap 95% CIs
| index | auc_ci_low | auc_ci_high | log_loss_ci_low | log_loss_ci_high | brier_ci_low | brier_ci_high |
|---|---|---|---|---|---|---|
| lr | 0.734 | 0.860 | 0.426 | 0.546 | 0.134 | 0.183 |
| lr_spline | 0.742 | 0.864 | 0.420 | 0.543 | 0.133 | 0.183 |
| rf | 0.717 | 0.838 | 0.448 | 0.561 | 0.145 | 0.190 |
| rf_cal | 0.725 | 0.845 | 0.440 | 0.565 | 0.141 | 0.192 |
| lgbm | 0.726 | 0.845 | 0.436 | 0.561 | 0.139 | 0.190 |
| lgbm_cal | 0.732 | 0.850 | 0.432 | 0.553 | 0.139 | 0.188 |
| lr_balanced | 0.738 | 0.864 | 0.489 | 0.596 | 0.160 | 0.206 |
| logit_unpenalised | 0.721 | 0.850 | 0.424 | 0.582 | 0.136 | 0.193 |

### Paired bootstrap differences vs LR
| index | auc_diff_vs_lr | auc_lo | auc_hi | log_loss_diff_vs_lr | log_loss_lo | log_loss_hi |
|---|---|---|---|---|---|---|
| lr_spline | 0.0052 | -0.0150 | 0.0251 | -0.0032 | -0.0187 | 0.0123 |
| rf | -0.0201 | -0.0597 | 0.0221 | 0.0192 | -0.0115 | 0.0475 |
| rf_cal | -0.0137 | -0.0507 | 0.0252 | 0.0174 | -0.0136 | 0.0466 |
| lgbm | -0.0109 | -0.0379 | 0.0194 | 0.0126 | -0.0093 | 0.0349 |
| lgbm_cal | -0.0071 | -0.0335 | 0.0220 | 0.0078 | -0.0128 | 0.0275 |
| lr_balanced | 0.0034 | -0.0024 | 0.0095 | 0.0594 | 0.0192 | 0.0985 |
| logit_unpenalised | -0.0112 | -0.0274 | 0.0048 | 0.0183 | -0.0079 | 0.0466 |

## Decisions: policies on the test set
| index | threshold | approval_rate | bad_rate_among_approved | defaults_caught | false_accepts | false_rejects | expected_cost_per_applicant |
|---|---|---|---|---|---|---|---|
| approve everyone | 0.500 | 1.000 | 0.300 | 0.000 | 75.000 | 0.000 | 1.500 |
| reject everyone | 0.500 | 0.000 | nan | 1.000 | 0.000 | 175.000 | 0.700 |
| model, naive cut-off 50% | 0.500 | 0.800 | 0.210 | 0.440 | 42.000 | 17.000 | 0.908 |
| model, cost-optimal cut-off 16.7% | 0.167 | 0.344 | 0.105 | 0.880 | 9.000 | 98.000 | 0.572 |

### Risk grades
| grade | n | mean_pd | observed_default | share |
|---|---|---|---|---|
| A (<10%) | 47 | 0.068 | 0.064 | 0.188 |
| B (10-20%) | 55 | 0.151 | 0.127 | 0.220 |
| C (20-30%) | 36 | 0.250 | 0.139 | 0.144 |
| D (30-50%) | 62 | 0.388 | 0.435 | 0.248 |
| E (>50%) | 50 | 0.630 | 0.660 | 0.200 |

### Cost-ratio sensitivity
| cost_ratio_FA:FR | threshold | approval_rate | bad_rate_among_approved | defaults_caught | false_accepts | false_rejects | expected_cost_per_applicant |
|---|---|---|---|---|---|---|---|
| 1:1 | 0.500 | 0.800 | 0.210 | 0.440 | 42 | 17 | 0.236 |
| 2:1 | 0.333 | 0.616 | 0.136 | 0.720 | 21 | 42 | 0.336 |
| 3:1 | 0.250 | 0.476 | 0.101 | 0.840 | 12 | 68 | 0.416 |
| 5:1 | 0.167 | 0.344 | 0.105 | 0.880 | 9 | 98 | 0.572 |
| 10:1 | 0.091 | 0.148 | 0.027 | 0.987 | 1 | 139 | 0.596 |
| 20:1 | 0.048 | 0.048 | 0.000 | 1.000 | 0 | 163 | 0.652 |

### Portfolio base-rate sensitivity
| assumed_portfolio_default_rate | mean_shifted_pd | approval_rate_at_16.7%_cutoff | approval_rate_unshifted |
|---|---|---|---|
| 0.030 | 0.042 | 0.964 | 0.344 |
| 0.050 | 0.068 | 0.908 | 0.344 |
| 0.100 | 0.124 | 0.728 | 0.344 |
| 0.200 | 0.220 | 0.476 | 0.344 |
| 0.300 | 0.304 | 0.344 | 0.344 |

### Three-way (approve / refer / decline, +-5pp band)
| band | n | observed_default |
|---|---|---|
| approve | 54 | 0.093 |
| decline | 141 | 0.447 |
| refer | 55 | 0.127 |

## Fairness: unawareness
| model_inputs | oof_auc | oof_log_loss | approval_rate@1/6 |
|---|---|---|---|
| all features | 0.786 | 0.497 | 0.341 |
| without protected attributes | 0.781 | 0.501 | 0.323 |

### Proxy test
| index | prevalence | proxy_auc |
|---|---|---|
| female | 0.310 | 0.704 |
| age<25 | 0.149 | 0.797 |
| age 50+ | 0.125 | 0.769 |
| non-foreign worker | 0.037 | 0.796 |

### Disparities (bootstrap 95% CI)
| attribute | group | metric | reference | value | ci_low | ci_high |
|---|---|---|---|---|---|---|
| sex | female | approval_rate_ratio | male | 0.705 | 0.558 | 0.867 |
| sex | female | false_reject_rate_diff | male | 0.103 | 0.027 | 0.184 |
| sex | female | calibration_gap_diff | male | -0.008 | -0.063 | 0.052 |
| age_group | <25 | approval_rate_ratio | 25-34 | 0.629 | 0.413 | 0.888 |
| age_group | <25 | false_reject_rate_diff | 25-34 | 0.119 | 0.009 | 0.231 |
| age_group | <25 | calibration_gap_diff | 25-34 | -0.021 | -0.102 | 0.067 |
| age_group | 35-49 | approval_rate_ratio | 25-34 | 1.464 | 1.192 | 1.781 |
| age_group | 35-49 | false_reject_rate_diff | 25-34 | -0.133 | -0.215 | -0.045 |
| age_group | 35-49 | calibration_gap_diff | 25-34 | 0.032 | -0.025 | 0.087 |
| age_group | 50+ | approval_rate_ratio | 25-34 | 1.693 | 1.333 | 2.114 |
| age_group | 50+ | false_reject_rate_diff | 25-34 | -0.213 | -0.316 | -0.099 |
| age_group | 50+ | calibration_gap_diff | 25-34 | -0.018 | -0.099 | 0.062 |
| foreign_worker | no | approval_rate_ratio | yes | 1.882 | 1.373 | 2.420 |
| foreign_worker | no | false_reject_rate_diff | yes | -0.236 | -0.409 | -0.056 |
| foreign_worker | no | calibration_gap_diff | yes | 0.077 | -0.027 | 0.162 |

## Limitations: selection-bias simulation
| model | training sample | evaluated on | n_train | n_eval | auc | mean_pd | observed_default | pd_bias(pred-obs) |
|---|---|---|---|---|---|---|---|---|
| LR (C=0.1) | trained on all applicants | whole test set | 750 | 250 | 0.801 | 0.304 | 0.300 | 0.004 |
| LR (C=0.1) | trained on all applicants | legacy-refused region | 750 | 83 | 0.706 | 0.504 | 0.542 | -0.039 |
| LR (C=0.1) | trained on legacy-accepted only | whole test set | 515 | 250 | 0.765 | 0.249 | 0.300 | -0.051 |
| LR (C=0.1) | trained on legacy-accepted only | legacy-refused region | 515 | 83 | 0.625 | 0.373 | 0.542 | -0.169 |
| LightGBM (fixed) | trained on all applicants | whole test set | 750 | 250 | 0.790 | 0.300 | 0.300 | -0.000 |
| LightGBM (fixed) | trained on all applicants | legacy-refused region | 750 | 83 | 0.691 | 0.510 | 0.542 | -0.032 |
| LightGBM (fixed) | trained on legacy-accepted only | whole test set | 515 | 250 | 0.769 | 0.244 | 0.300 | -0.056 |
| LightGBM (fixed) | trained on legacy-accepted only | legacy-refused region | 515 | 83 | 0.653 | 0.373 | 0.542 | -0.169 |

## Figures
![baseline_forest](figures/baseline_forest.png)
![decision_cost_curve](figures/decision_cost_curve.png)
![eda_default_rates](figures/eda_default_rates.png)
![eda_information_value](figures/eda_information_value.png)
![eda_numeric_distributions](figures/eda_numeric_distributions.png)
![explain_dependence](figures/explain_dependence.png)
![explain_importance](figures/explain_importance.png)
![explain_local_highest_pd](figures/explain_local_highest_pd.png)
![explain_local_lowest_pd](figures/explain_local_lowest_pd.png)
![explain_local_near_cutoff](figures/explain_local_near_cutoff.png)
![fairness_groups](figures/fairness_groups.png)
![limits_learning_curve](figures/limits_learning_curve.png)
![test_discrimination](figures/test_discrimination.png)
![test_reliability](figures/test_reliability.png)
