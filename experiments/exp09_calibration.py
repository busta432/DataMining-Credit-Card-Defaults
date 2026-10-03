"""WP13 / exp09 -- calibration.

Mandatory rather than optional: WP14's cost analysis converts probabilities into
decisions through Elkan's threshold, and uncalibrated probabilities invalidate that.

Each fold splits its own training half into a fit portion and a calibration portion, so
the calibrator never sees the fold's validation rows and the test set is untouched.
Reliability curves are serialised so WP19 can plot without refitting.

    python -m experiments.exp09_calibration
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS, SEED
from src.data import RAW_FEATURES, training_context
from src.metrics import reliability
from src.models import scale_pos_weight, xgb_classifier
from src.runner import Timer, fit_one, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"
SPW_PARAMS = ARTIFACTS / "best_params_spw.json"
CALIB_SIZE = 0.20
N_BINS = 10


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params_a = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]
        # exp03 writes the spw-arm parameters; fall back to arm A's if it has not run.
        params_c = (json.loads(SPW_PARAMS.read_text(encoding="utf-8"))["params"]
                    if SPW_PARAMS.exists() else params_a)

        ctx = training_context(RAW_FEATURES)
        X, y, folds = ctx["X"], ctx["y"], ctx["folds"]
        spw = scale_pos_weight(y)

        arms = {
            "A_spw1": (params_a, 1.0),
            "C_spw": (params_c, spw),
        }
        methods = ["raw", "sigmoid", "isotonic"]

        # oof[arm][method] accumulates one probability per training row.
        oof = {a: {m: np.full(len(y), np.nan) for m in methods} for a in arms}

        for arm, (params, arm_spw) in arms.items():
            for tr, va in folds:
                i_fit, i_cal = train_test_split(tr, test_size=CALIB_SIZE,
                                                stratify=y[tr], random_state=SEED)
                base = fit_one(xgb_classifier(params, spw=arm_spw),
                               X.iloc[i_fit], y[i_fit], seed=SEED)
                oof[arm]["raw"][va] = base.predict_proba(X.iloc[va])[:, 1]
                for m in ("sigmoid", "isotonic"):
                    # sklearn 1.9 removed cv="prefit"; FrozenEstimator is the replacement,
                    # so `fit` below trains the calibration map only, never the booster.
                    cal = CalibratedClassifierCV(FrozenEstimator(base), method=m)
                    cal.fit(X.iloc[i_cal], y[i_cal])
                    oof[arm][m][va] = cal.predict_proba(X.iloc[va])[:, 1]

        rows, curves = [], {}
        for arm in arms:
            for m in methods:
                p = oof[arm][m]
                assert not np.isnan(p).any(), f"{arm}/{m} left rows unpredicted"
                label = f"{arm}_{m}"
                cv = {
                    "roc_auc_mean": float(roc_auc_score(y, p)),
                    "roc_auc_sd": None,
                    "pr_auc_mean": float(average_precision_score(y, p)),
                    "pr_auc_sd": None,
                    "brier_mean": float(brier_score_loss(y, p)),
                    "brier_sd": None,
                    "n_fits": len(folds),
                }
                curves[label] = reliability(y, p, n_bins=N_BINS)
                # Expected calibration error over the same bins.
                ece = float(sum(b["n"] * abs(b["mean_pred"] - b["frac_pos"])
                                for b in curves[label] if b["n"] > 0) / len(y))
                rows.append(make_row(label, cv, params=arms[arm][0],
                                     n_features=X.shape[1],
                                     extra={"arm": arm, "method": m, "ece": ece,
                                            "spw": arms[arm][1],
                                            "mean_predicted": float(p.mean())}))
                print(f"{label:<20} brier {cv['brier_mean']:.5f}  ece {ece:.5f}  "
                      f"roc_auc {cv['roc_auc_mean']:.5f}  mean_p {p.mean():.4f}")
                np.save(ARTIFACTS / f"oof_calib_{label}.npy", p)

        by = {r["label"]: r for r in rows}
        prevalence = float(y.mean())
        a_raw, c_raw = by["A_spw1_raw"], by["C_spw_raw"]
        best = min(rows, key=lambda r: r["cv"]["brier_mean"])

        notes = (
            f"Prevalence is {prevalence:.4f}. The spw=1 model predicts a mean probability "
            f"of {a_raw['mean_predicted']:.4f} -- essentially the base rate -- with Brier "
            f"{a_raw['cv']['brier_mean']:.5f} and ECE {a_raw['ece']:.5f}. Reweighting to "
            f"spw={spw:.2f} inflates the mean prediction to {c_raw['mean_predicted']:.4f} "
            f"and worsens Brier to {c_raw['cv']['brier_mean']:.5f} (ECE "
            f"{c_raw['ece']:.5f}), because scale_pos_weight shifts the score distribution "
            f"without changing the underlying event rate. That is the concrete cost of "
            f"reweighting that ROC-AUC cannot see. "
            f"Best Brier overall: {best['label']} at {best['cv']['brier_mean']:.5f}. "
            f"Arm B of exp03 is omitted here because it is arm A's fitted model read at a "
            f"different threshold -- its probabilities are identical, which is precisely "
            f"why threshold-moving is preferable to reweighting when probabilities are "
            f"consumed downstream. "
            f"Reliability curves ({N_BINS} equal-width bins) are serialised in config for "
            f"WP19."
        )
        save_result("exp09_calibration", "Calibration: Brier, ECE and reliability curves",
                    rows, notes, t.seconds,
                    config={"n_bins": N_BINS, "calib_size": CALIB_SIZE,
                            "prevalence": prevalence, "scale_pos_weight": spw,
                            "reliability_curves": curves,
                            "params_c_source": ("exp03" if SPW_PARAMS.exists()
                                                else "fallback to arm A params")})


if __name__ == "__main__":
    main()
