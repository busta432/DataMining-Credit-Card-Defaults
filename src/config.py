"""Single source of truth for seeds, paths, cost constants and search spaces."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

for _d in (FIGDIR, RESULTS, ARTIFACTS):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------- randomness

SEED = 42
# Distinct from SEED so the repeated-CV folds used for variance reporting are not
# correlated with the folds the tuner selected against.
CV_SEED = 7

N_FOLDS = 5
N_REPEATS = 5
TEST_SIZE = 0.20
# Fraction carved out of each training fold as an early-stopping eval_set, so the
# CV validation fold is never seen by early stopping.
ES_SLICE = 0.15
ES_ROUNDS = 50

# ---------------------------------------------------------------- cost model
# FN (missed default) = LGD x EAD. Retail unsecured loss-given-default ~0.70.
LGD = 0.70
# FP (rejected good customer) = forgone net interest margin ~0.12 x exposure.
NIM = 0.12
# Headline cost ratio C_FN / C_FP = 0.70 / 0.12 = 5.83 -> reported as 6:1.
R_HEADLINE = 6
R_GRID = [1, 2, 5, 6, 10, 20]
THRESHOLD_GRID_N = 501

# ---------------------------------------------------------------- xgboost

# enable_categorical defaults to True in xgboost 3.4, and shap 0.52 reads that flag
# alone to decide a model has categorical splits -- which blocks interventional
# TreeSHAP. Numeric models must set it False explicitly. See SETUP.md.
XGB_FIXED = {
    "tree_method": "hist",
    "enable_categorical": False,
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": SEED,
}

# Declarative so tuning.py can suggest from it and the report can print it.
# (kind, low, high, log)
SEARCH_SPACE = {
    "max_depth": ("int", 3, 8, False),
    "eta": ("float", 0.01, 0.30, True),
    "min_child_weight": ("float", 1.0, 20.0, True),
    "subsample": ("float", 0.60, 1.0, False),
    "colsample_bytree": ("float", 0.50, 1.0, False),
    "gamma": ("float", 0.0, 5.0, False),
    "reg_lambda": ("float", 1e-2, 100.0, True),
    "reg_alpha": ("float", 1e-3, 10.0, True),
}

# Early stopping decides the real count; this is only a ceiling.
N_ESTIMATORS_CAP = 2000
# The plan budgeted 60-80 trials against an estimated 20-40 min. Measured cost is ~2.3 s
# per trial, so the budget was raised: more TPE trials is strictly better and 150 still
# finishes in ~6 min. Recorded in the exp02 notes as a deviation.
N_TRIALS = 150

# ---------------------------------------------------------------- imbalance arms
# scale_pos_weight is fixed per arm rather than tuned: ROC-AUC is invariant to
# monotone score transformations, so the tuner would be blind to it and pick
# arbitrarily. Fixing it per arm turns the imbalance study into a controlled
# comparison instead of a confound.
IMBALANCE_ARMS = {
    "A_baseline":        {"spw": 1.0, "threshold": "fixed_0.5", "sampler": None},
    "B_threshold":       {"spw": 1.0, "threshold": "oof_cost",  "sampler": None},
    "C_spw":             {"spw": None, "threshold": "fixed_0.5", "sampler": None},
    "C_spw_threshold":   {"spw": None, "threshold": "oof_cost",  "sampler": None},
    "D_smote":           {"spw": 1.0, "threshold": "oof_cost",  "sampler": "smote"},
}

TUNING_METRIC = "roc_auc"
