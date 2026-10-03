"""WP6 / exp02 -- Optuna TPE hyperparameter search.

Produces artifacts/best_params.json and artifacts/final_model.json, which every
downstream experiment reads. Reports the tuning gain against XGBoost's library
defaults (rung 5 of the complexity ladder) on identical folds.

    python -m experiments.exp02_tuning
"""

from __future__ import annotations

import json

import numpy as np

from src.config import ARTIFACTS, N_TRIALS, SEARCH_SPACE, SEED
from src.data import RAW_FEATURES, training_context
from src.models import xgb_classifier, xgb_library_defaults
from src.runner import Timer, cv_evaluate, fit_one, make_row, save_result
from src.tuning import run_study

BEST_PARAMS = ARTIFACTS / "best_params.json"
FINAL_MODEL = ARTIFACTS / "final_model.json"
STUDY_DB = ARTIFACTS / "optuna_study.db"


def main():
    with Timer() as t:
        ctx = training_context(RAW_FEATURES)
        X, y, folds, rfolds = ctx["X"], ctx["y"], ctx["folds"], ctx["repeated_folds"]

        print(f"tuning on {len(X):,} training rows, {X.shape[1]} features, "
              f"{N_TRIALS} trials, {len(folds)}-fold ROC-AUC objective")
        study = run_study(X, y, folds, STUDY_DB, n_trials=N_TRIALS, seed=SEED)
        best = study.best_trial
        params = dict(best.params)
        print(f"best trial #{best.number}: ROC-AUC {best.value:.5f}")
        print(json.dumps(params, indent=2))

        # Refit the selected configuration over the 25 repeated-CV folds for the
        # mean +/- SD the report quotes. The tuner itself never saw these folds.
        tuned = cv_evaluate(X, y, lambda: xgb_classifier(params), rfolds, seed=SEED)
        default = cv_evaluate(X, y, xgb_library_defaults, rfolds, seed=SEED)
        gain = tuned["cv"]["roc_auc_mean"] - default["cv"]["roc_auc_mean"]
        pooled_sd = float(np.hypot(tuned["cv"]["roc_auc_sd"], default["cv"]["roc_auc_sd"]))

        # Final model: one fit on the whole training portion, early stopping against an
        # inner slice of it. This is the model WP14-WP18 explain and evaluate.
        final = fit_one(xgb_classifier(params), X, y, seed=SEED)
        n_trees = int(final.best_iteration) + 1
        final.get_booster().save_model(FINAL_MODEL)

        BEST_PARAMS.write_text(json.dumps({
            "params": params,
            "n_estimators_final": n_trees,
            "best_iteration": int(final.best_iteration),
            "cv_roc_auc_tuning_folds": float(best.value),
            "trial_number": int(best.number),
            "seed": SEED,
            "features": RAW_FEATURES,
        }, indent=2), encoding="utf-8")
        print(f"saved -> {BEST_PARAMS}\nsaved -> {FINAL_MODEL}  ({n_trees} trees)")

        n_complete = len([tr for tr in study.trials if tr.state.name == "COMPLETE"])
        n_pruned = len([tr for tr in study.trials if tr.state.name == "PRUNED"])

        rows = [
            make_row("tuned", tuned["cv"], params=params, n_features=X.shape[1],
                     extra={"n_trees": n_trees,
                            "best_iteration_mean": tuned.get("best_iteration_mean")}),
            make_row("library_defaults", default["cv"], params={},
                     n_features=X.shape[1], extra={"n_trees": 100}),
        ]

        notes = (
            f"Optuna TPE, {n_complete} complete and {n_pruned} pruned trials, MedianPruner. "
            f"Trial budget raised from the planned 60-80 to {N_TRIALS}: measured cost was "
            f"~2.3 s per trial rather than the estimated 15-30 s, so a wider search was free. "
            f"Tuning gain over library defaults on the 25 repeated-CV fits: "
            f"{gain:+.5f} ROC-AUC against a pooled fold-SD of {pooled_sd:.5f} "
            f"({'larger than' if abs(gain) > pooled_sd else 'smaller than'} one pooled SD). "
            f"scale_pos_weight was excluded from the search space by design: ROC-AUC is "
            f"invariant to monotone score transformations, so the tuner cannot see it. "
            f"Early stopping used an inner 15% slice of each training fold, never the "
            f"validation fold and never the test set."
        )
        save_result("exp02_tuning", "Hyperparameter search (Optuna TPE)", rows, notes,
                    t.seconds,
                    config={"n_trials": N_TRIALS, "objective": "roc_auc",
                            "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
                            "tuning_gain_roc_auc": gain, "pooled_fold_sd": pooled_sd,
                            "n_complete": n_complete, "n_pruned": n_pruned})


if __name__ == "__main__":
    main()
