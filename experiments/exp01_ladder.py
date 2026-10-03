"""WP5 / exp01 -- the complexity ladder. Answers research question 1.

Each rung adds modelling capability at a measurable interpretability cost, so the
accuracy-versus-explainability trade-off becomes a measured axis rather than an
assertion. Rung 1 also recovers the PAY_0-alone ROC-AUC benchmark that was lost with
the uncommitted Phase 1 multivariate notebook.

Rungs 2 and 3 are untuned internal reference implementations used to locate XGBoost on
a capability axis. They are NOT the group's Logistic Regression or Decision Tree
submissions, which other members tune separately.

    python -m experiments.exp01_ladder
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ARTIFACTS
from src.data import RAW_FEATURES, training_context
from src.models import count_tree_nodes, ladder_specs
from src.runner import Timer, cv_evaluate, fit_one, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(
                f"MISSING ARTIFACT: {BEST_PARAMS}\n"
                "Produced by WP6. Run: python -m experiments.exp02_tuning"
            )
        best = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES)
        X_all, y, rfolds = ctx["X"], ctx["y"], ctx["repeated_folds"]
        prevalence = float(y.mean())

        rows = []
        for spec in ladder_specs(best_params=best):
            X = X_all[spec["features"]]
            out = cv_evaluate(X, y, spec["factory"], rfolds)
            oof = out["oof"]

            # Decisions at the naive 0.5 threshold, to make the accuracy trap explicit.
            pred = (oof >= 0.5).astype(int)
            acc = float((pred == y).mean())
            rec = float(pred[y == 1].mean())

            # Interpretability cost: the number of parameters or rules a human must read
            # to reproduce one decision. None means not directly available.
            cost = spec["interp_cost"]
            if cost is None:
                fitted = fit_one(spec["factory"](), X, y)
                cost = count_tree_nodes(fitted)

            rows.append(make_row(
                spec["label"], out["cv"],
                params=best if spec["label"] == "rung6_xgb_tuned" else {},
                n_features=X.shape[1],
                extra={
                    "model": spec["model"],
                    "interpretability": spec["interpretability"],
                    "interp_cost": cost,
                    "accuracy_at_0.5": acc,
                    "recall_at_0.5": rec,
                },
            ))
            print(f"{spec['label']:<20} roc_auc {out['cv']['roc_auc_mean']:.4f} "
                  f"+/- {out['cv']['roc_auc_sd']:.4f}   acc@0.5 {acc:.4f}  "
                  f"recall@0.5 {rec:.4f}  interp_cost {cost}")

        # Rung 1b: PAY_0 used directly as an ordinal score, with no model fitted at all.
        # The depth-1 tree above binarises PAY_0 into two groups and so throws the
        # ordinal gradient away; ranking on the raw code keeps it. The gap between the
        # two is the price of insisting on a single readable rule, and it is this row --
        # not the tree -- that reproduces the Phase 1 benchmark.
        pay0 = X_all["PAY_0"].to_numpy(dtype=float)
        score_auc = [roc_auc_score(y[va], pay0[va]) for _, va in rfolds]
        score_pr = [average_precision_score(y[va], pay0[va]) for _, va in rfolds]
        rows.insert(2, make_row(
            "rung1b_pay0_score",
            {"roc_auc_mean": float(np.mean(score_auc)),
             "roc_auc_sd": float(np.std(score_auc, ddof=1)),
             "pr_auc_mean": float(np.mean(score_pr)),
             "pr_auc_sd": float(np.std(score_pr, ddof=1)),
             "brier_mean": None, "brier_sd": None, "n_fits": len(rfolds)},
            params={}, n_features=1,
            extra={"model": "PAY_0 as an ordinal score (no model fitted)",
                   "interpretability": "rank on one column",
                   "interp_cost": 1,
                   "accuracy_at_0.5": None, "recall_at_0.5": None}))
        print(f"{'rung1b_pay0_score':<20} roc_auc {np.mean(score_auc):.4f} "
              f"+/- {np.std(score_auc, ddof=1):.4f}   (no model: ranking on the raw code)")

        by = {r["label"]: r for r in rows}
        rule_auc = by["rung1_pay0_rule"]["cv"]["roc_auc_mean"]
        score_mean = by["rung1b_pay0_score"]["cv"]["roc_auc_mean"]
        tuned_auc = by["rung6_xgb_tuned"]["cv"]["roc_auc_mean"]
        tuned_sd = by["rung6_xgb_tuned"]["cv"]["roc_auc_sd"]
        forest_auc = by["rung4_forest"]["cv"]["roc_auc_mean"]
        default_auc = by["rung5_xgb_default"]["cv"]["roc_auc_mean"]

        notes = (
            f"Rung 0 reaches {by['rung0_majority']['accuracy_at_0.5']:.4%} accuracy with "
            f"zero recall, which is the accuracy trap in one line: the majority-class rate "
            f"is 1 - prevalence = {1 - prevalence:.4%}. "
            f"Rung 1, a depth-1 tree on PAY_0, reaches only {rule_auc:.4f} -- short of the "
            f"0.690 Phase 1 reported from the notebook that was never committed. The "
            f"discrepancy is informative rather than an error: a single split binarises "
            f"PAY_0 into two groups and discards the ordinal gradient, so rung 1b ranks on "
            f"the raw code with no model at all and reaches {score_mean:.4f}, recovering "
            f"the benchmark. The {score_mean - rule_auc:+.4f} between them is the measured "
            f"price of insisting on one readable rule, and it is the first rung of the "
            f"interpretability cost this experiment exists to quantify. "
            f"Tuned XGBoost reaches {tuned_auc:.4f} +/- {tuned_sd:.4f}, a gain of "
            f"{tuned_auc - rule_auc:+.4f} over a single readable rule and "
            f"{tuned_auc - default_auc:+.4f} over untuned XGBoost. Bagging alone "
            f"(rung 4) reaches {forest_auc:.4f}, so boosting contributes "
            f"{tuned_auc - forest_auc:+.4f} beyond bagging. "
            f"Rungs 2 and 3 are untuned internal references, not the group's separately "
            f"tuned Logistic Regression and Decision Tree submissions. "
            f"Interpretability cost counts parameters or nodes a human must read to "
            f"reproduce one decision; for the ensembles it is total node count across all "
            f"trees, which is why it is reported as 'not directly available - see SHAP'."
        )
        save_result("exp01_ladder", "Complexity ladder: capability against interpretability",
                    rows, notes, t.seconds,
                    config={"prevalence": prevalence, "n_rungs": len(rows),
                            "cv": "RepeatedStratifiedKFold(5x5, random_state=7)"})


if __name__ == "__main__":
    main()
