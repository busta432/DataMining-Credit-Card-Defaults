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
