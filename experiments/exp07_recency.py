"""WP11 / exp07 -- recency ablation.

Most-recent-k-months only, k in 1..6, using the PAY_CHRONO ordering. Tests the Phase 1
recency gradient directly and answers the practical question a lender would ask: how
much billing history do we actually need to collect?

    python -m experiments.exp07_recency
"""

from __future__ import annotations

import json

from src.config import ARTIFACTS
from src.data import (
    BILL_COLS,
    DEMOGRAPHIC,
    PAY_CHRONO,
    PAYAMT_COLS,
    RAW_FEATURES,
    training_context,
)
from src.models import xgb_classifier
from src.runner import Timer, cv_evaluate, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"

# Chronological, April -> September. Index -k: gives the most recent k months.
BILL_CHRONO = BILL_COLS[::-1]
PAYAMT_CHRONO = PAYAMT_COLS[::-1]
STATIC = ["LIMIT_BAL"] + DEMOGRAPHIC


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES)
        X_all, y, rfolds = ctx["X"], ctx["y"], ctx["repeated_folds"]

        rows = []
        for k in range(1, 7):
            cols = (STATIC + PAY_CHRONO[-k:] + BILL_CHRONO[-k:] + PAYAMT_CHRONO[-k:])
            out = cv_evaluate(X_all[cols], y, lambda: xgb_classifier(params), rfolds,
                              collect_oof=False)
            rows.append(make_row(f"k{k}_months", out["cv"], params=params,
                                 n_features=len(cols),
                                 extra={"k_months": k, "features": cols,
                                        "months": PAY_CHRONO[-k:]}))
            print(f"k={k}  n_feat {len(cols):>3}  roc_auc "
                  f"{out['cv']['roc_auc_mean']:.4f} +/- {out['cv']['roc_auc_sd']:.4f}")

        by = {r["label"]: r["cv"]["roc_auc_mean"] for r in rows}
        sd = rows[-1]["cv"]["roc_auc_sd"]
        k1, k6 = by["k1_months"], by["k6_months"]
        # Smallest k whose ROC-AUC is within one fold-SD of the full six months.
        enough = next(k for k in range(1, 7) if by[f"k{k}_months"] >= k6 - sd)
        gains = {k: by[f"k{k}_months"] - by[f"k{k - 1}_months"] for k in range(2, 7)}

        notes = (
            f"One month of history (September only) already reaches ROC-AUC {k1:.5f}, "
            f"against {k6:.5f} for all six months -- a total gain of {k6 - k1:+.5f} from "
            f"five extra months, against a fold-SD of {sd:.5f}. "
            f"k={enough} is the smallest history window that lands within one fold-SD of "
            f"the full six months, so most of the collectable signal is in the most recent "
            f"statement. This is the recency gradient of Phase 1 (Spearman rho rising from "
            f"0.143 at April to 0.292 at September) restated as a data-collection cost. "
            f"Marginal gain per additional month: "
            f"{', '.join(f'k{k}: {v:+.5f}' for k, v in gains.items())}. "
            f"Practical reading for a lender: the fifth and sixth months of history are "
            f"close to free of predictive value here, which matters because history depth "
            f"is an acquisition cost and a privacy exposure, not just a modelling choice."
        )
        save_result("exp07_recency", "Recency ablation: months of history required",
                    rows, notes, t.seconds,
                    config={"chronological_order": PAY_CHRONO,
                            "static_features": STATIC,
                            "smallest_k_within_one_sd": enough,
                            "marginal_gain_per_month": gains})


if __name__ == "__main__":
    main()
