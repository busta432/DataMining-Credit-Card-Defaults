"""WP9 / exp05 -- engineered-feature benchmark. Promised in Phase 1 section 2.2.2.

Phase 1 committed to benchmarking aggregate features rather than assuming they help,
and predicted that aggregation may *lose* signal by collapsing the recency gradient.
This tests that prediction.

    python -m experiments.exp05_engineered
"""

from __future__ import annotations

import json

import numpy as np

from src.config import ARTIFACTS
from src.data import ENGINEERED, RAW_FEATURES, training_context
from src.metrics import paired_fold_test
from src.models import xgb_classifier
from src.runner import Timer, cv_evaluate, make_row, per_fold_metric, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES + ENGINEERED, engineered=True)
        X_all, y, folds, rfolds = ctx["X"], ctx["y"], ctx["folds"], ctx["repeated_folds"]

        # RFE to 23 features, so the "raw + engineered" arm is compared at equal width
        # rather than simply being given more columns.
        from sklearn.feature_selection import RFE

        selector = RFE(xgb_classifier(params, early_stopping=False),
                       n_features_to_select=len(RAW_FEATURES), step=3)
        selector.fit(X_all, y)
        rfe_cols = list(X_all.columns[selector.support_])
        print(f"RFE kept {len(rfe_cols)} of {X_all.shape[1]}: {rfe_cols}")

        arms = [
            ("raw_23", RAW_FEATURES),
            ("raw_plus_engineered", RAW_FEATURES + ENGINEERED),
            ("engineered_only", ENGINEERED),
            ("rfe_top23", rfe_cols),
        ]

        rows, folds_auc = [], {}
        for label, cols in arms:
            Xa = X_all[cols]
            out = cv_evaluate(Xa, y, lambda: xgb_classifier(params), rfolds,
                              collect_oof=False)
            folds_auc[label] = per_fold_metric(out)
            rows.append(make_row(label, out["cv"], params=params, n_features=len(cols),
                                 extra={"features": cols,
                                        "n_engineered": len([c for c in cols
                                                             if c in ENGINEERED])}))
            print(f"{label:<22} n_feat {len(cols):>3}  roc_auc "
                  f"{out['cv']['roc_auc_mean']:.4f} +/- {out['cv']['roc_auc_sd']:.4f}")

        base = folds_auc["raw_23"]
        tests = {k: paired_fold_test(v - base) for k, v in folds_auc.items() if k != "raw_23"}
        by = {r["label"]: r["cv"]["roc_auc_mean"] for r in rows}
        sd = rows[0]["cv"]["roc_auc_sd"]
        d_add = by["raw_plus_engineered"] - by["raw_23"]
        d_only = by["engineered_only"] - by["raw_23"]
        verdict = ("confirmed" if d_only < -sd else
                   "refuted" if d_only > sd else "neither confirmed nor refuted")

        notes = (
            f"Adding the 17 engineered features to the raw 23 moves ROC-AUC by "
            f"{d_add:+.5f} against a fold-SD of {sd:.5f}. Replacing the raw columns "
            f"entirely with the engineered ones moves it by {d_only:+.5f}. "
            f"Phase 1 predicted aggregation would lose the recency gradient; on this "
            f"evidence that prediction is {verdict} at the one-SD level. "
            f"RFE to {len(rfe_cols)} features from the combined pool provides an "
            f"equal-width comparison, so the raw-plus-engineered arm is not simply "
            f"rewarded for having more columns. "
            f"PAY_RATIO_n is NaN where the statement balance is zero or negative "
            f"(1,930 rows carry a negative BILL_AMT per Phase 1 section 1.6); XGBoost "
            f"learns a default direction for missing values natively rather than "
            f"requiring imputation, which is why NaN was preferred to a sentinel."
        )
        save_result("exp05_engineered", "Engineered-feature benchmark", rows, notes,
                    t.seconds,
                    config={"engineered": ENGINEERED, "rfe_selected": rfe_cols,
                            "paired_fold_tests_vs_raw_23": tests})


if __name__ == "__main__":
    main()
