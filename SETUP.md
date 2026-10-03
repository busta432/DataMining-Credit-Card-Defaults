# Setup and reproduction

How to recreate the environment and re-run the analysis from a fresh clone.

## What is and is not in this repository

| Committed | Not committed | Why not |
|---|---|---|
| `data_mining_assignment.ipynb` | `.venv/` | ~561 MB, 21,510 files, platform-specific binaries, hardcoded absolute paths |
| `figures/` (16 PNGs) | `Decisions.md` | Working document, not a submission artefact |
| `requirements.txt` | `PROJECT_TRACKER.md` | Working document |
| `.gitignore`, `SETUP.md` | `CLAUDE.md` | Tooling config |

The notebook is committed **with its outputs**, so the analysis can be read without running
anything. The trade-off is a ~1.6 MB file whose diffs are noisy, because output images are
stored inline as base64.

## Recreate the environment

Requires **Python 3.14**.

```bash
git clone https://github.com/busta432/DataMining-Credit-Card-Defaults.git
cd DataMining-Credit-Card-Defaults

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The dataset is **not** stored in the repository — it is downloaded at runtime from the UCI
Machine Learning Repository by `ucimlrepo` (id=350), so the first run needs an internet
connection.

### Phase 2 additions

Phase 2 adds `xgboost`, `shap`, `optuna`, `imbalanced-learn` and `statsmodels`. All five install
and import on **Python 3.14.3** — `numba 0.67.0` publishes `cp314` wheels, so the `shap`
dependency chain needs no version fallback. Verified versions:

| Package | Version |
|---|---|
| xgboost | 3.4.1 |
| shap | 0.52.0 |
| optuna | 5.0.0 |
| imbalanced-learn | 0.14.2 |
| statsmodels | 0.15.0 |
| numba / llvmlite | 0.67.0 / 0.49.0 |

Two environment gotchas were hit and resolved:

**1. `MAX_PATH` on Windows.** Installing `ipykernel` into a `.venv` inside the nested course
directory fails on `jedi`'s bundled `django-stubs` type stubs, which exceed the 260-character
path limit. Either enable long-path support, or create the venv at a short path:

```bash
python -m venv "$HOME/venvs/dm314"
"$HOME/venvs/dm314/Scripts/pip" install -r requirements.txt
```

**2. `enable_categorical` defaults to `True` in xgboost 3.4.** `shap` 0.52 reads that flag
alone — not whether any feature is actually of `category` dtype — and refuses interventional
TreeSHAP with `NotImplementedError: Categorical split is not yet supported`. Every numeric
model in `src/models.py` therefore passes `enable_categorical=False` explicitly. Only the
`enable_categorical=True` arm of the encoding ablation sets it, and that arm is not explained
with interventional TreeSHAP.

## Phase 2 layout

Phase 1 is one notebook. Phase 2 is a package plus a thin narrative notebook, because fourteen
experiments in a single kernel is neither restartable nor cacheable, and because the assignment
requires a source-code appendix that reads better as modules than as a 3,000-line notebook.

```
src/          library: config, data, metrics, costs, models, tuning, runner, plotting, report
experiments/  one runnable script per experiment, exp01..exp14, plus aggregate.py and figures.py
results/      one JSON per experiment + manifest.csv        (committed)
artifacts/    splits, tuned params, fitted models, OOF and test probabilities, Optuna DB  (gitignored)
figures/      Phase 1 PNGs + Phase 2 PNGs, prefixed p2_
```

Two rules make the numbers reconcile. All evaluation goes through `runner.cv_evaluate()`, so no
experiment writes its own CV loop. And every number printed in the report is read back out of
`results/`, never retyped — `src/report.py` builds each table from the JSON.

`artifacts/` is gitignored because fitted models are large and regenerable. `results/` **is**
committed, so the report's numbers survive without a re-run.

### Run one experiment

Each script is idempotent and standalone:

```bash
"$HOME/venvs/dm314/Scripts/python" -m experiments.exp03_imbalance
```

Scripts declare their inputs. An experiment whose required artifact is missing fails immediately
and names the experiment that produces it, rather than silently recomputing it under a different
seed — that is how reported numbers stop reconciling.

The dependency order is: `exp02_tuning` first (it writes `artifacts/best_params.json`, which
almost everything else reads), then `exp01` and `exp04`–`exp08` in any order, then `exp03`,
then `exp09`–`exp13`, and `exp14_final` last. `exp14` is the only experiment licensed to touch
the hold-out set.

### Measured runtimes

Wall clock on the reference machine (Windows 11, 8 physical cores, `tree_method="hist"`,
23,955 training rows). These are the `runtime_sec` fields of `results/*.json`, so they are
measurements rather than estimates.

| Script | What it does | Minutes |
|---|---|---|
| `exp01_ladder` | Complexity ladder, 8 rungs × 25 fits | 1.7 |
| `exp02_tuning` | Optuna TPE, 150 trials × 5 folds | 19.4 |
| `exp03_imbalance` | 4 imbalance arms, each tuned separately | 5.0 |
| `exp04_encoding` | Encoding ablation, 4 arms | 3.1 |
| `exp05_engineered` | Engineered-feature benchmark, 4 arms | 2.8 |
| `exp06_groups` | Feature-group ablation, 11 arms | 5.0 |
| `exp07_recency` | Recency ablation, k = 1..6 months | 3.9 |
| `exp08_learning_curve` | 5 fractions × 5 seeds | 2.0 |
| `exp09_calibration` | Brier, ECE, reliability curves | 0.3 |
| `exp10_cost` | Cost sweep — reads cached OOF probabilities, fits nothing | <0.1 |
| `exp11_shap` | TreeSHAP on 2,000 held-out rows | 0.4 |
| `exp12_stability` | 5 reseeded refits + 5 importance measures | 1.3 |
| `exp13_fairness` | Subgroup metrics with Wilson intervals | 0.1 |
| `exp14_final` | 10-seed spread, PAY_0-dropped refit, hold-out, DeLong | 2.5 |
| | **Total, run serially** | **≈48** |

`exp02_tuning` is 41% of the total on its own. It resumes from `artifacts/optuna_study.db`, so
an interrupted search does not restart from trial 1. Waves 3 and 4 (`exp04`–`exp13`) only read
`best_params.json` and can be run in parallel if the cores are available.

### Rebuild the tables and figures

```bash
"$HOME/venvs/dm314/Scripts/python" -m experiments.aggregate   # results/manifest.csv
"$HOME/venvs/dm314/Scripts/python" -m experiments.figures     # figures/p2_*.png
```

Both read `results/` and `artifacts/` only. Neither fits a model, so both finish in seconds.

### Rebuild the Word deliverables

```bash
"$HOME/venvs/dm314/Scripts/python" report/build_docx.py
```

Converts the two Markdown sources into the submitted `.docx` files:

| Source | Output |
|---|---|
| `report/phase2_report.md` | `report/phase2_report.docx` |
| `report/annotated_bibliography.md` | `report/annotated_bibliography.docx` |

Markdown remains the source of truth — edit the `.md`, never the `.docx`, then re-run the
build. Requires `python-docx`; `matplotlib` rasterises the two display equations into
`figures/_eq/` because `python-docx` cannot emit OMML. Inline maths is mapped to Unicode
through an explicit `MATH` table in the script, so an unmapped expression raises rather than
leaking LaTeX into the document.

Layout is pinned in the script so the page budget is reproducible: US Letter, 0.75 in margins,
Calibri 10 pt body, 8.5 pt tables, Consolas 8.5 pt code, body figures 4.4 in wide. Under these
settings the assessed body (Introduction → References) occupies **9 pages** against the 10-page
cap; the appendix runs to page 28 and is explicitly uncapped by the brief.

To confirm the page count after editing, on a machine with Word installed:

```powershell
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$d = $w.Documents.Open("$PWD\report\phase2_report.docx", $false, $true)
$d.Repaginate(); $r = $d.Content; $r.Find.Text = 'Appendix A'
if ($r.Find.Execute()) { "body = " + ($r.Information(1) - 1) + " pages" }
$d.Close($false); $w.Quit()
```

## Run the notebook

**In VS Code:** open `data_mining_assignment.ipynb`, then select the kernel via
*Select Kernel → Python Environments → `.venv`*.

> If cells fail with `Running cells with '.conda' requires the ipykernel package`, VS Code is
> pointed at the wrong interpreter. Select `.venv` explicitly — this is the single most common
> failure on this project.

**From the CLI**, executing every cell and rewriting the outputs in place:

```bash
./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 data_mining_assignment.ipynb
```

Takes roughly 1–2 minutes. It regenerates all 16 PNGs in `figures/`.

### The Phase 2 notebook

`phase2_xgboost.ipynb` renders the Phase 2 narrative. It computes nothing — it reads `results/`
and displays `figures/` — so it executes in well under a minute and will run on a machine that
never fitted a model, provided `results/` and `figures/` are present:

```bash
"$HOME/venvs/dm314/Scripts/python" -m jupyter nbconvert --to notebook --execute --inplace \
  phase2_xgboost.ipynb
```

Measured: 36 cells, 0 errors, **5.5 seconds**. Invoke nbconvert through `python -m` rather than
the bare `jupyter` command — the console entry point is not always on `PATH` inside the venv,
and the module form works either way.

If it raises `FileNotFoundError` naming a `results/*.json`, the corresponding experiment has not
been run.

## Verifying it worked

The notebook contains 11 inline `assert` statements that fail loudly if the data or the
cleaning logic changes. A clean run means all of these held:

- Raw data is 30,000 × 24 with 6,636 defaults and no nulls
- 108 rows sit in 52 duplicate groups, 21 with contradictory labels
- After cleaning: **29,944** rows, **6,622** defaults
- `EDUCATION ⊆ {1,2,3,4}` and `MARRIAGE ⊆ {1,2,3}`

To confirm no cell errored:

```bash
./.venv/bin/python -c "
import nbformat as nbf
nb = nbf.read('data_mining_assignment.ipynb', as_version=4)
errs = [o for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error']
print(f'{len(nb.cells)} cells, {len(errs)} errors')
"
```

## Updating figures only

Figures are written by the `save_fig()` helper defined in the setup cell, at 150 dpi into
`figures/`. Re-running the notebook overwrites them all; there is no separate figure build step.
