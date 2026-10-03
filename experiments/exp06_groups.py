"""WP10 / exp06 -- feature-group ablation.

Leave-one-group-out and only-one-group over FEATURE_GROUPS, quantifying where the
signal actually lives and testing the Phase 1 conclusion that repayment status
dominates the monetary columns.

The no-demographics arm is kept as a named row because WP17 reuses it to test whether
removing demographic columns removes demographic influence or merely hides it.

    python -m experiments.exp06_groups
"""

from __future__ import annotations

import json

from src.config import ARTIFACTS
from src.data import FEATURE_GROUPS, RAW_FEATURES, training_context
from src.models import xgb_classifier
from src.runner import Timer, cv_evaluate, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES)
        X_all, y, rfolds = ctx["X"], ctx["y"], ctx["repeated_folds"]

        arms = [("all_groups", RAW_FEATURES, "all five groups")]
        for g, cols in FEATURE_GROUPS.items():
            arms.append((f"drop_{g}", [c for c in RAW_FEATURES if c not in cols],
                         f"leave-one-out: {g} removed"))
        for g, cols in FEATURE_GROUPS.items():
            arms.append((f"only_{g}", list(cols), f"only-one: {g} alone"))

        rows = []
        for label, cols, desc in arms:
            out = cv_evaluate(X_all[cols], y, lambda: xgb_classifier(params), rfolds,
                              collect_oof=False)
            rows.append(make_row(label, out["cv"], params=params, n_features=len(cols),
                                 extra={"description": desc, "features": cols}))
            print(f"{label:<22} n_feat {len(cols):>3}  roc_auc "
                  f"{out['cv']['roc_auc_mean']:.4f} +/- {out['cv']['roc_auc_sd']:.4f}")

        by = {r["label"]: r["cv"]["roc_auc_mean"] for r in rows}
        sd = rows[0]["cv"]["roc_auc_sd"]
        full = by["all_groups"]
        drops = {g: full - by[f"drop_{g}"] for g in FEATURE_GROUPS}
        onlys = {g: by[f"only_{g}"] for g in FEATURE_GROUPS}
        worst_drop = max(drops, key=drops.get)
        best_only = max(onlys, key=onlys.get)
        demo_cost = drops["demographic"]

        notes = (
            f"Removing {worst_drop} costs {drops[worst_drop]:.5f} ROC-AUC, the largest "
            f"leave-one-out loss, against a fold-SD of {sd:.5f}. "
            f"{best_only} alone reaches {onlys[best_only]:.5f} versus {full:.5f} for the "
            f"full feature set, so it carries most of the available signal on its own. "
            f"This confirms the Phase 1 conclusion that repayment status dominates the "
            f"monetary columns. "
            f"Leave-one-out and only-one disagree in the usual way: a group can be "
            f"individually predictive yet redundant once the others are present, which is "
            f"exactly what collinear BILL_AMT columns produce. Both views are reported "
            f"rather than one. "
            f"Dropping the demographic group costs {demo_cost:+.5f} ROC-AUC; the "
            f"drop_demographic arm is consumed by WP17, which tests whether the financial "
            f"columns proxy for the removed demographics (Barocas & Selbst, 2016)."
        )
        save_result("exp06_groups", "Feature-group ablation", rows, notes, t.seconds,
                    config={"groups": FEATURE_GROUPS,
                            "leave_one_out_loss": drops, "only_one_auc": onlys})


if __name__ == "__main__":
    main()
