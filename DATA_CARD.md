# Data card: Statlog (German Credit Data)

## Source and licence
* **Origin:** Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg; donated 1994 to the UCI Machine Learning Repository, dataset id 144 (<https://archive.ics.uci.edu/dataset/144>). UCI lists the dataset under CC BY 4.0 - confirm on the dataset page before redistributing.
* **Size:** 1,000 loans, 20 application-time attributes (7 numeric / ordinal, 13 categorical), 1 outcome.
* **Obtaining it:** `pdrisk.data.load_raw()` uses `data/raw/german.data` if its checksum matches; otherwise it downloads from the UCI zip (primary) or a GitHub mirror (fallback) and **refuses any file whose SHA-256 differs** from the pinned value `ec12a88b...c017`. The file is also committed to this repo (it is ~80 KB).
* **Citation:** Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.

## What the rows are
One row per consumer loan that **was granted** by a (south) German bank. Reported collection period 1973-1975 (stated in the later "South German Credit" documentation, not in the original file), so amounts are **1970s Deutsche Mark** and the credit environment is not today's.

## Target definition (the part that is easy to get wrong)
* UCI codes the outcome `1 = good`, `2 = bad`. We define **`default = 1` iff class = 2** (bad). PD is therefore `P(default = 1 | application data)`. Row 1 is a good loan (default = 0), row 2 a bad one (default = 1); 300 of 1,000 rows are defaults. This polarity is covered by a unit test.
* **What "bad" means is not documented** - there is no stated default definition (e.g. 90 days past due), no observation window / performance horizon and no application date. Consequently: no point-in-time or out-of-time validation is possible and PDs cannot be labelled "12-month PD" or "lifetime PD". We call them "PD for the loan's outcome as labelled by the bank".
* **Leakage check:** all 20 attributes are application-time facts (balances, history, purpose, demographics). `status` (checking-account status) has a very high information value (0.6, conventionally "suspicious" above 0.5). We keep it because the variable is an application-time balance band, but note that if "status" were recorded *after* a loan went bad (e.g. an overdrawn account observed at default) it would be leakage - the documentation does not rule this out.

## Provenance checks (why we decode the raw codes ourselves)
Three copies of the data were compared row-by-row (numeric columns and the target agree in 100% of rows across all three):

| copy | finding |
|---|---|
| Original UCI-coded file (`A11`, `A34`, ...) | Reference. Category frequencies match the UCI documentation. |
| Pre-decoded GitHub CSV (selva86/datasets) | **Wrong labels:** `status` code A11 labelled "... < 100 DM" instead of "< 0 DM"; **`purpose` labels shifted** for 8 of 10 codes (e.g. 280 'radio/television' loans labelled "domestic appliances", 97 'business' labelled "others"). Other columns correct. |
| `scorecardpy` bundled copy | `status` / `purpose` labels agree with the original, but **`personal_status_sex` is scrambled** relative to the original codes (cross-tab is near-independent although the marginals are identical). |

Using either decoded copy unchecked would silently corrupt interpretation (and the sex-based fairness analysis). We therefore decode the original codes with the official codebook (`pdrisk.data.CODEBOOK`) and test the decoded marginals against UCI frequencies (`tests/test_data.py`). Residual risk: the codebook was transcribed by us; the UCI-frequency tests and the three-way cross-check are the evidence it is right.

## Known quirks and limitations of the data
* **Sampling / base rate:** a 30% bad rate is far above typical retail default rates (low single digits to ~10%). The sample is almost certainly not representative of the bank's through-the-door population (the documentation does not describe the sampling). PDs are therefore calibrated to *this sample*, not to a portfolio; see `prior_shift` and the base-rate sensitivity table.
* **Selection bias:** only accepted applicants have outcomes (no rejected applicants, no reject inference).
* **Cost matrix:** UCI provides one: approving a bad loan costs 5, rejecting a good one costs 1. We use it as the default; real LGD/EAD/margin would replace it.
* **Combined attribute:** `personal_status_sex` mixes marital status and sex; code A95 ("female: single") has zero rows, so "female" only appears as *divorced/separated/married* (n=310). Sex is *derived* (male/female) for the audit only.
* **Ordinal codes:** `installment_rate`, `present_residence`, `number_credits`, `people_liable` are small integer codes (1-4 or 1-2) as supplied; we treat them as numeric ordinals. The direction of the band coding is as in the original file.
* **Small n:** 1,000 rows and 300 defaults; 225 defaults in the development split. Group sizes for fairness are tiny (e.g. 37 non-foreign workers).
* **Protected attributes** (age, sex-in-personal-status, foreign-worker) are present. Their use in real credit decisions is restricted in the EU/Germany (this project is not legal advice).

## Data dictionary
| column | meaning |
|---|---|
| `status` | Status of existing checking account (balance band) |
| `duration` | Loan duration in months |
| `credit_history` | Credit history with this and other lenders |
| `purpose` | Purpose of the loan |
| `amount` | Credit amount in Deutsche Mark (DM) |
| `savings` | Savings account / bonds balance band |
| `employment_duration` | Present employment since |
| `installment_rate` | Instalment rate as % of disposable income, ordinal code 1-4 (as coded by UCI) |
| `personal_status_sex` | Personal status and sex (combined attribute; sex is NOT recorded separately) |
| `other_debtors` | Other debtors / guarantors |
| `present_residence` | Present residence since, ordinal code 1-4 (as coded by UCI) |
| `property` | Most valuable property owned |
| `age` | Age in years |
| `other_installment_plans` | Other instalment plans (bank/stores/none) |
| `housing` | Housing situation |
| `number_credits` | Number of existing credits at this bank (1-4) |
| `job` | Job category |
| `people_liable` | Number of people liable to provide maintenance (1 or 2) |
| `telephone` | Telephone registered in customer's name |
| `foreign_worker` | Foreign worker |
| `default` | TARGET. 1 = bad credit / default, 0 = good credit (UCI class 2 -> 1, class 1 -> 0) |
