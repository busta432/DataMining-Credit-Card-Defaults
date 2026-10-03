"""WP7 / exp03 -- class-imbalance strategy comparison.

Four strategies, each tuned independently so `scale_pos_weight` is a controlled factor
rather than a confound (see src/tuning.py for why it is excluded from the search space).

Arm A is the spw=1 configuration already tuned by exp02; arm B is that same fitted model
read at a cost-minimising threshold instead of 0.5. Arms C and D get their own studies.

Writes artifacts/oof_proba_<label>.npy for WP13 (calibration) and WP14 (cost).

    python -m experiments.exp03_imbalance
"""

from __future__ import annotations

import json

import numpy as np

from src.config import ARTIFACTS, N_TRIALS, R_HEADLINE, SEED
from src.costs import cost_at_threshold, pick_threshold_oof
from src.data import RAW_FEATURES, training_context
from src.metrics import evaluate
from src.models import SmoteXGB, scale_pos_weight, xgb_classifier
from src.runner import Timer, cv_evaluate, make_row, save_result
from src.tuning import run_study

BEST_PARAMS = ARTIFACTS / "best_params.json"
SMOTE_TRIALS = 60  # each trial resamples to ~2x rows, so the budget is trimmed


def _oof(X, y, factory, folds, label):
    out = cv_evaluate(X, y, factory, folds)
    np.save(ARTIFACTS / f"oof_proba_{label}.npy", out["oof"])
    return out


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(
                f"MISSING ARTIFACT: {BEST_PARAMS}\n"
                "Produced by WP6. Run: python -m experiments.exp02_tuning"
            )
        params_a = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES)
        X, y, folds, rfolds = ctx["X"], ctx["y"], ctx["folds"], ctx["repeated_folds"]
        spw = scale_pos_weight(y)
        print(f"scale_pos_weight (neg/pos) = {spw:.4f}")

        # --- arm C: independently tuned at spw = neg/pos
        study_c = run_study(X, y, folds, ARTIFACTS / "optuna_study_spw.db",
                            study_name="xgb_spw", n_trials=N_TRIALS, spw=spw, seed=SEED)
        params_c = dict(study_c.best_trial.params)
        print(f"arm C best tuning-fold ROC-AUC {study_c.best_value:.5f}")

        # --- arm D: independently tuned with SMOTE inside fit
        study_d = run_study(
            X, y, folds, ARTIFACTS / "optuna_study_smote.db",
            study_name="xgb_smote", n_trials=SMOTE_TRIALS, seed=SEED,
            model_fn=lambda p, s, sd: SmoteXGB(params=p, seed=sd),
        )
        params_d = dict(study_d.best_trial.params)
        print(f"arm D best tuning-fold ROC-AUC {study_d.best_value:.5f}")

        # WP13 (calibration) and WP17 (fairness) read the reweighted arm's parameters.
        (ARTIFACTS / "best_params_spw.json").write_text(json.dumps(
            {"params": params_c, "spw": spw,
             "cv_roc_auc_tuning_folds": float(study_c.best_value)}, indent=2),
            encoding="utf-8")
        (ARTIFACTS / "best_params_smote.json").write_text(json.dumps(
            {"params": params_d, "sampler": "SMOTE",
             "cv_roc_auc_tuning_folds": float(study_d.best_value)}, indent=2),
            encoding="utf-8")

        configs = {
            "A": (lambda: xgb_classifier(params_a), params_a),
            "C": (lambda: xgb_classifier(params_c, spw=spw), params_c),
            "D": (lambda: SmoteXGB(params=params_d), params_d),
        }

        # Variance reporting over the 25 repeated-CV fits, and OOF probabilities from
        # the 5 tuning folds (which partition the training rows exactly).
        cv, oof = {}, {}
        for key, (factory, _) in configs.items():
            cv[key] = cv_evaluate(X, y, factory, rfolds, collect_oof=False)
            oof[key] = _oof(X, y, factory, folds, {"A": "A_baseline",
                                                   "C": "C_spw",
                                                   "D": "D_smote"}[key])["oof"]
            print(f"arm {key}: roc_auc {cv[key]['cv']['roc_auc_mean']:.4f} "
                  f"+/- {cv[key]['cv']['roc_auc_sd']:.4f}  "
                  f"brier {cv[key]['cv']['brier_mean']:.4f}")

        tau = {}
        for key in configs:
            tau[key], _ = pick_threshold_oof(y, oof[key], r=R_HEADLINE)
        print("cost-minimising OOF thresholds:",
              {k: round(v, 4) for k, v in tau.items()})

        def decision_extras(key, threshold):
            m = evaluate(y, oof[key], threshold)
            return {
                "oof_threshold": threshold,
                "oof_recall": m["recall"],
                "oof_precision": m["precision"],
                "oof_f1": m["f1"],
                "oof_cost_r6": cost_at_threshold(y, oof[key], threshold, r=R_HEADLINE),
                "spw": 1.0 if key in ("A", "D") else spw,
                "sampler": "SMOTE" if key == "D" else None,
            }

        rows = [
            make_row("A_baseline", cv["A"]["cv"], params=params_a, n_features=X.shape[1],
                     extra=decision_extras("A", 0.5)),
            make_row("B_threshold", cv["A"]["cv"], params=params_a, n_features=X.shape[1],
                     extra={**decision_extras("A", tau["A"]),
                            "note": "identical fitted model to A_baseline; only the "
                                    "decision threshold differs, so CV ranking and "
                                    "calibration metrics are the same by construction"}),
            make_row("C_spw", cv["C"]["cv"], params=params_c, n_features=X.shape[1],
                     extra=decision_extras("C", 0.5)),
            make_row("C_spw_threshold", cv["C"]["cv"], params=params_c,
                     n_features=X.shape[1], extra=decision_extras("C", tau["C"])),
            make_row("D_smote", cv["D"]["cv"], params=params_d, n_features=X.shape[1],
                     extra=decision_extras("D", tau["D"])),
        ]

        auc_a = cv["A"]["cv"]["roc_auc_mean"]
        auc_c = cv["C"]["cv"]["roc_auc_mean"]
        auc_d = cv["D"]["cv"]["roc_auc_mean"]
        sd_a = cv["A"]["cv"]["roc_auc_sd"]
        br_a = cv["A"]["cv"]["brier_mean"]
        br_c = cv["C"]["cv"]["brier_mean"]

        notes = (
            f"Reweighting changes ranking by {auc_c - auc_a:+.5f} ROC-AUC against a "
            f"fold-SD of {sd_a:.5f}, so A and C are not distinguishable on ranking. "
            f"Calibration does move: Brier {br_a:.5f} (spw=1) versus {br_c:.5f} "
            f"(spw={spw:.2f}), a {br_c - br_a:+.5f} change. This is a mechanistic result, "
            f"not a null one: reweighting and threshold-moving are two parameterisations "
            f"of the same decision shift, but only threshold-moving preserves probability "
            f"semantics, and the cost analysis in WP14 consumes probabilities. "
            f"SMOTE reaches {auc_d:.5f} ({auc_d - auc_a:+.5f} versus A) and is rejected on "
            f"dataset-specific grounds rather than generic ones: interpolating two clients "
            f"produces values such as PAY_0 = 0.37, a repayment code that does not exist, "
            f"and Phase 1 established PAY_n is non-monotonic and categorical-like. "
            f"max_delta_step was considered and dismissed -- it addresses imbalance beyond "
            f"roughly 100:1, and this dataset is {spw:.1f}:1. "
            f"Thresholds were frozen from out-of-fold training predictions at r=6; the test "
            f"set is not touched by this experiment."
        )
        save_result("exp03_imbalance", "Class-imbalance strategy comparison", rows, notes,
                    t.seconds,
                    config={"scale_pos_weight": spw, "r": R_HEADLINE,
                            "thresholds": {k: tau[k] for k in tau},
                            "n_trials_spw": N_TRIALS, "n_trials_smote": SMOTE_TRIALS,
                            "arm_A_params_source": "exp02 (spw=1 study)"})


if __name__ == "__main__":
    main()
