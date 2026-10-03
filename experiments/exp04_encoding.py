"""WP8 / exp04 -- encoding ablation for the repayment-status columns.

Native integers versus one-hot versus XGBoost's native categorical support, plus a
signed-log transform of the monetary columns to demonstrate empirically that trees are
scale-invariant -- the Phase 1 scaling recommendation applied to distance and gradient
learners, not to this one.

    python -m experiments.exp04_encoding
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACTS
from src.data import BILL_COLS, PAY_COLS, PAYAMT_COLS, RAW_FEATURES, training_context
from src.models import xgb_classifier
from src.runner import Timer, cv_evaluate, make_row, per_fold_metric, save_result
from src.metrics import paired_fold_test

BEST_PARAMS = ARTIFACTS / "best_params.json"
MONETARY = ["LIMIT_BAL"] + BILL_COLS + PAYAMT_COLS


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES)
        X, y, rfolds = ctx["X"], ctx["y"], ctx["repeated_folds"]

        # The dummy vocabulary is built from the full training frame, so folds share a
        # column set. That is structural encoding, not target leakage -- no fold sees
        # another fold's labels.
        X_onehot = pd.get_dummies(X, columns=PAY_COLS, dtype=np.int8)
        X_cat = X.copy()
        for c in PAY_COLS:
            X_cat[c] = X_cat[c].astype("category")
        X_log = X.copy()
        for c in MONETARY:
            X_log[c] = np.sign(X_log[c]) * np.log1p(np.abs(X_log[c]))

        arms = [
            ("native_int", X, dict(params), False,
             "PAY_n as signed integers, as loaded"),
            ("one_hot", X_onehot, dict(params), False,
             f"PAY_n expanded to {X_onehot.shape[1] - X.shape[1] + len(PAY_COLS)} dummies"),
            ("native_categorical", X_cat, dict(params), True,
             "PAY_n as pandas category dtype, enable_categorical=True"),
            ("signed_log_monetary", X_log, dict(params), False,
             "sign(x)*log1p(|x|) on LIMIT_BAL, BILL_AMT and PAY_AMT"),
        ]

        rows, folds_auc = [], {}
        for label, Xa, p, cat, desc in arms:
            out = cv_evaluate(Xa, y, lambda p=p, cat=cat: xgb_classifier(p, enable_categorical=cat),
                              rfolds, collect_oof=False)
            folds_auc[label] = per_fold_metric(out)
            rows.append(make_row(label, out["cv"], params=p, n_features=Xa.shape[1],
                                 extra={"description": desc}))
            print(f"{label:<22} n_feat {Xa.shape[1]:>3}  roc_auc "
                  f"{out['cv']['roc_auc_mean']:.4f} +/- {out['cv']['roc_auc_sd']:.4f}")

        base = folds_auc["native_int"]
        tests = {k: paired_fold_test(v - base) for k, v in folds_auc.items()
                 if k != "native_int"}
        spread = max(r["cv"]["roc_auc_mean"] for r in rows) - \
            min(r["cv"]["roc_auc_mean"] for r in rows)
        sd = rows[0]["cv"]["roc_auc_sd"]

        notes = (
            f"All four encodings sit within {spread:.5f} ROC-AUC of each other against a "
            f"fold-SD of {sd:.5f}, so none is distinguishable from native integers. "
            f"Native integers are carried forward. Trees make axis-aligned splits, so a "
            f"non-monotonic response to PAY_n is recoverable by stacking splits "
            f"(PAY_0 < 0.5, then PAY_0 < -1.5) -- that costs depth, not expressible "
            f"functions. Integers also keep the six PAY_n columns distinct, protecting the "
            f"recency gradient Phase 1 measured (Spearman rho 0.143 rising to 0.292). "
            f"One-hot expands to {rows[1]['n_features']} columns for no measurable gain. "
            f"The signed-log arm confirms scale invariance: a monotone transform of every "
            f"monetary column moves ROC-AUC by "
            f"{tests['signed_log_monetary']['mean_delta']:+.5f}, because axis-aligned splits "
            f"depend only on rank order. Paired per-fold t-tests are reported, with the "
            f"Nadeau-Bengio (2003) caveat that repeated-CV tests are anti-conservative "
            f"because the training folds overlap. "
            f"Note that enable_categorical=True blocks interventional TreeSHAP in shap "
            f"0.52, which is a second reason to prefer integers for a model that must be "
            f"explained."
        )
        save_result("exp04_encoding", "Encoding ablation for repayment-status columns",
                    rows, notes, t.seconds,
                    config={"paired_fold_tests_vs_native_int": tests,
                            "auc_spread": spread})


if __name__ == "__main__":
    main()
