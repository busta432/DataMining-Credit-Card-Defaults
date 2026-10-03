# DataMining-Credit-Card-Defaults

Exploratory data analysis and modelling proposal for the **Default of Credit Card Clients**
dataset (Taiwan, 2005) — [UCI ML Repository id=350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

30,000 credit card clients, 23 explanatory variables, and a binary target: whether the client
defaulted on their next monthly payment.

## The research angle

Beyond predicting default, this project asks an interpretability question: **can a model
trained *without* demographic inputs still reconstruct them from financial variables?**

If credit limit, bill amounts and payment history carry enough information to recover sex or
education level, then dropping those columns does not actually remove their influence — it
only hides it. Phase 1 establishes the demographic baseline this question depends on; later
phases test whether the information leak is real and, if so, where in a network it lives.


## Phase 1 findings

The dataset is complete — no missing values anywhere — but semantically messy, and several of
its problems are not mentioned in the official data dictionary.

**The repayment codes are only half documented.** `PAY_n` is defined as `-1 = paid duly` and
`1..9 = months delayed`, but the data also contains `-2` and `0` — and `0` is the *modal value
in every month*, covering roughly half of all client-months. These are read here as
`-2` = dormant, `-1` = paid in full, `0` = revolving credit.

**The response to repayment status is non-monotonic.** Default rises by 56 percentage points
between "revolving" (12.8%) and "two months late" (69.2%). But among the non-delinquent codes
the ordering is counterintuitive: clients who paid *in full* default more often (16.8%) than
those revolving a balance (12.8%). The ordering is also unstable across months, so it is
reported as an anomaly rather than a law — and it means `PAY_n` must not be treated as a
continuous ordinal variable in a linear model.

![Default rate by repayment status](figures/biv_default_by_pay0.png)

**21 duplicate groups carry contradictory labels** — identical values across all 23 features
but different outcomes. No classifier can separate these, so they establish a small
irreducible error floor.

**Repayment code `1` is nearly absent before September** — 0, 0, 2, 4, 28 occurrences in
April–August, then 3,688 in September. A behavioural variable does not jump two orders of
magnitude in its final month, which suggests `PAY_0` may be constructed differently from the
five historical columns. Since it is also the most predictive variable in the dataset, this is
recorded as a limitation.

**Default rates differ measurably across demographic groups**, which is the baseline the
later interpretability work builds on.

![Default rate by demographic group](figures/biv_default_by_demographic.png)

**Modelling implications carried forward:**

| Finding | Consequence |
|---|---|
| 22.1% default prevalence | Accuracy is unusable — a majority-class classifier scores 77.9%; evaluate with precision–recall |
| Skewness up to 30.4 on payment columns | Scaling required; robust (median/IQR) scaler |
| Non-monotonic `PAY_n` response | Encode categorically, or use a non-linear learner |
| Predictive strength rises with recency | The six `PAY_n` columns are not exchangeable |
| Contradictory duplicate records | A non-zero Bayes error floor exists |

## Phase 2 findings — XGBoost

Phase 2 implements **XGBoost only** (an individual deliverable; the Decision Tree and Logistic
Regression are submitted separately by other group members). Fourteen experiments, all committed
to `results/*.json`. Every metric below is mean ± SD over the same 25 repeated-CV fits, and
**nothing inside one fold-SD is claimed as an improvement**.

**The complexity ladder answers the proposal's first research question.** Each rung buys
capability at a reading cost measured in parameters or nodes.

| Rung | ROC-AUC | Cost to read |
|---|---|---|
| Majority class | 0.5000 ± 0.0000 | 1 |
| `PAY_0`, one rule | 0.6441 ± 0.0075 | 1 |
| `PAY_0` as a raw score | 0.6910 ± 0.0083 | 1 |
| Logistic regression | 0.7248 ± 0.0086 | 24 |
| Decision tree | 0.7542 ± 0.0103 | 109 |
| Random forest | 0.7641 ± 0.0096 | 692,674 |
| XGBoost, defaults | 0.7619 ± 0.0080 | 100 |
| **XGBoost, tuned** | **0.7845 ± 0.0071** | 499 |

![Complexity ladder](figures/p2_fig1_complexity_ladder.png)

**The missing Phase 1 benchmark was recovered, and the discrepancy was informative.** Phase 1
reported a `PAY_0`-alone ROC-AUC of 0.690 from a notebook that was never committed. A depth-1
tree on `PAY_0` reaches only 0.6441, because one split binarises the column and discards its
ordinal gradient. Ranking on the raw code with *no model at all* reaches **0.6910**, recovering
the benchmark. The +0.0469 between them is the measured price of insisting on one readable rule.

**Accuracy cannot distinguish any of these models.** From the one-line rule to tuned XGBoost,
accuracy at the default threshold moves only between 0.8094 and 0.8197 while ROC-AUC moves from
0.6441 to 0.7845. The majority-class classifier scores 77.88% with zero recall.

| Finding | Consequence |
|---|---|
| Tuning gains **+0.0226** AUC (~2.1 pooled SD) but the whole 150-trial search spans <0.01 | Most of the gain is early stopping and a low `eta`, not locating a precise optimum |
| Signed-log monetary transform scores **identically** to raw (0.7845 ± 0.0071 both) | Trees are scale-invariant; the Phase 1 scaling advice applies to other learners, not this one |
| One-hot, native-integer and `enable_categorical` all within 0.0017 | Native integers carried forward — fewer columns, recency gradient preserved, SHAP readable |
| Dropping the six `PAY_n` costs **−0.0471**; those six alone reach 0.7492 | Repayment status dominates: 95.5% of the AUC from 26% of the features |
| Dropping all demographics costs **−0.0013** (inside one SD) | Demographics are nearly free to remove — which makes the fairness question interesting, not moot |
| Engineered aggregates alone lose **−0.0059**; RFE preferred raw ingredients | Phase 1's prediction that aggregation discards the recency gradient is **confirmed** |
| One month of history reaches 0.7669 of the six-month 0.7833 | A lender could halve its history requirement for no measurable loss |
| Untuned XGBoost (0.7619) does **not** beat the random forest (0.7641) | Boosting at defaults buys nothing over bagging — the advantage here is tuning, not architecture |
| The forest needs **692,674 nodes**; tuned XGBoost needs **499** for a higher score | The real XGBoost win is capability per unit of opacity |

**Every intervention aimed at the class imbalance failed, and the failures are the result.**
At 22.11% prevalence the dataset is mildly imbalanced — enough to break accuracy, not enough to
break learning.

| Arm | ROC-AUC | Brier | OOF cost at r = 6 |
|---|---|---|---|
| A — `spw` = 1, τ = 0.5 | 0.7845 ± 0.0071 | 0.1348 | 21,331 |
| B — same model, τ = 0.1341 from OOF | 0.7845 ± 0.0071 | 0.1348 | **14,239** |
| C — `spw` = 3.52, τ = 0.5 | 0.7848 ± 0.0074 | 0.1801 | 15,431 |
| C′ — `spw` = 3.52, τ = 0.3580 | 0.7848 ± 0.0074 | 0.1801 | 14,215 |
| D — SMOTE inside the fold | 0.7681 ± 0.0088 | 0.2200 | — |

Reweighting moves ROC-AUC by **+0.0003 against a fold-SD of 0.0071** — one twenty-third of a
standard deviation — while inflating the mean predicted probability from 0.2189 to 0.4299 and
ECE from 0.0212 to **0.2088**. Arms B and C′ reach within 0.17% of the same cost by different
routes; only one of them keeps the probabilities interpretable. Reweighting is threshold-moving
performed indirectly, with the probabilities damaged on the way.

| Finding | Consequence |
|---|---|
| Empirical cost-optimal τ = 0.1341 against Elkan's analytic 0.1429; MAE across the sweep falls 0.0141 → 0.0079 after calibration | Probabilities are calibrated well enough for the cost analysis to be valid — the confirming direction |
| Threshold selection cuts expected cost **33.2%**; the entire 150-trial search bought +0.0226 AUC | The decision rule, not the model, is where the money is |
| SHAP reproduces the Phase 1 `PAY_0` non-monotonicity from the model's internals, ρ = **0.886** | Convergent validation — the model learned the EDA finding without being told it |
| SHAP additive to **3.7e-06**; rank Spearman ρ̄ **0.9375**, Jaccard@10 **0.8727** under reseeding | Stable enough to be an audit trail — except inside the collinear `BILL_AMT` block |
| `weight` importance ranks `PAY_0` **twelfth** while SHAP ranks it first (cover–weight ρ = **−0.526**) | The built-in importance plot would have inverted the central finding |
| A model that cannot see `SEX` still reproduces **58.0%** of the selection-rate gap | Fairness through unawareness hides influence rather than removing it |
| The model amplifies: a 3.44 pp observed gap becomes an **8.72 pp** selection gap | Disparity has to be measured on decisions, not on inputs |
| Hold-out ROC-AUC **0.7829**, recall 0.8444 at the frozen τ, 33.5% cost saving vs τ = 0.5 | The OOF threshold transferred; no test-set tuning anywhere |
| DeLong vs the single rule: **+0.1399** [0.1268, 0.1531], *p* = 1.5e-96 | The capability gain is real and far larger than fold noise |
| 10-seed spread 0.7820 ± 0.0053, with one seed at 0.7671 | Single-seed results in this range are not reliable to three decimals |
| **0** contradictory duplicate groups survive cleaning | The expected Bayes-floor argument was withdrawn; the ceiling claim rests on the learning curve instead |

Full write-up in [`report/phase2_report.md`](report/phase2_report.md); narrative and figures in
[`phase2_xgboost.ipynb`](phase2_xgboost.ipynb).

## Repository layout

```
data_mining_assignment.ipynb   Phase 1 analysis, 49 cells, executes top-to-bottom
phase2_xgboost.ipynb           Phase 2 narrative; reads results/, computes nothing
src/                           Library: data, metrics, costs, models, tuning, runner, plotting
experiments/                   exp01..exp14, one runnable script each, + aggregate and figures
results/                       One JSON per experiment + manifest.csv (committed)
artifacts/                     Splits, tuned params, models, probabilities (gitignored)
figures/                       Phase 1 + Phase 2 figures at 150 dpi
requirements.txt               Pinned dependencies (Python 3.14)
SETUP.md                       Environment setup and reproduction guide
```

Phase 2 is a package rather than a second notebook because fourteen experiments in one kernel
is neither restartable nor cacheable. `results/` is committed so every number in the report
survives without a re-run; `artifacts/` is not, because fitted models are large and
regenerable.

The dataset is **not** stored here — `ucimlrepo` downloads it at runtime, so the first run
needs an internet connection. The virtualenv is not committed either; `requirements.txt`
reproduces it.

Most of the decisions made so far include inline commentary in the notebook's markdown cells, with what has motivated the decision.

## Quick start

```bash
git clone https://github.com/busta432/DataMining-Credit-Card-Defaults.git
cd DataMining-Credit-Card-Defaults

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Then open the notebook and select the `.venv` interpreter as the kernel, or execute it
headlessly:

```bash
./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 data_mining_assignment.ipynb
```

Full details, including how to verify a clean run, are in [SETUP.md](SETUP.md).

## Method

Analysis follows the **CRISP-DM** process model, profiling the raw data *before* any cleaning
so that each cleaning rule is argued from evidence gathered beforehand — and so that anomalies
are not erased before they can be found. 

The notebook carries 11 inline assertions covering
row counts, class balance and post-cleaning category domains, so any silent change in the data
or the cleaning logic fails loudly.

## References List

- Yeh, I.-C. and Lien, C.-H. (2009) 'The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients', *Expert Systems with
  Applications*, 36(2), pp. 2473–2480. — source paper for this dataset
- Chapman, P. *et al.* (2000) *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- Barocas, S. and Selbst, A.D. (2016) 'Big data's disparate impact', *California Law Review*,
  104(3), pp. 671–732.
- He, H. and Garcia, E.A. (2009) 'Learning from imbalanced data', *IEEE TKDE*, 21(9),
  pp. 1263–1284.
- Saito, T. and Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the
  ROC plot when evaluating binary classifiers on imbalanced datasets', *PLoS ONE*, 10(3),
  e0118432.
- Tukey, J.W. (1977) *Exploratory data analysis*. Reading, MA: Addison-Wesley.
