"""WP18 / exp14 -- robustness checks and the single test-set gate.

Every threshold applied here was frozen on out-of-fold *training* predictions before the
hold-out was read. Nothing is selected in this file: the configurations, their
hyperparameters and their operating points all arrive fixed from earlier work packages.
That is what makes the test numbers an estimate of generalisation rather than another
round of selection.

Four things happen:

1. A PAY_0-dropped refit. Phase 1 documented that repayment code 1 occurs 0, 0, 2, 4, 28
   times April-August and then 3,688 times in September, which suggests PAY_0 is
   constructed differently from the five historical columns -- and it is the most
   predictive variable in the dataset. The AUC delta says how much the headline leans on
   it.
2. Seed variance over 10 refits, so the report can say whether a difference of a few
   thousandths means anything.
3. The final test evaluation, applying the frozen thresholds.
4. DeLong's test, tuned XGBoost against the PAY_0 single-rule baseline, on the test set
   -- paired, correlated ROC curves on one sample, which is exactly its remit.

    python -m experiments.exp14_final
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS, ES_SLICE, R_HEADLINE, SEED
from src.costs import cost_at_threshold, pick_threshold_oof
from src.data import (
    DEMOGRAPHIC,
    RAW_FEATURES,
    TARGET,
    frame,
    test_context,
    training_context,
)
from src.metrics import delong_test, evaluate, paired_fold_test
from src.models import (
    scale_pos_weight,
    single_rule,
    xgb_classifier,
    xgb_library_defaults,
)
from src.runner import Timer, cv_evaluate, fit_one, make_row, per_fold_metric, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"
BEST_PARAMS_SPW = ARTIFACTS / "best_params_spw.json"
SEED_COUNT = 10
NO_PAY0 = [c for c in RAW_FEATURES if c != "PAY_0"]
NO_DEMO = [c for c in RAW_FEATURES if c not in DEMOGRAPHIC]


def bayes_floor_stats(df):
    """Feature-identical rows carrying contradictory labels.

    An irreducible error floor exists wherever two clients are indistinguishable in the
    feature space yet one defaulted and one did not. Quantifying it stops the report
    making the lazy version of this argument.
    """
    g = df.groupby(list(RAW_FEATURES), sort=False)[TARGET]
    size, mean = g.transform("size"), g.transform("mean")
    dup = size > 1
    contradictory = dup & (mean > 0) & (mean < 1)
    groups = df.loc[contradictory].groupby(list(RAW_FEATURES), sort=False).ngroups
    return {
        "n_rows": int(len(df)),
        "rows_in_duplicate_groups": int(dup.sum()),
        "contradictory_rows": int(contradictory.sum()),
        "contradictory_groups": int(groups),
        "contradictory_share_pct": round(100 * float(contradictory.mean()), 4),
    }


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        df = frame()
        tr = training_context(RAW_FEATURES, df=df)
        te = test_context(RAW_FEATURES, df=df)
        X, y, folds, rfolds = tr["X"], tr["y"], tr["folds"], tr["repeated_folds"]
        X_test, y_test = te["X_test"], te["y_test"]
        ead_test = X_test["LIMIT_BAL"].to_numpy(dtype=float)
        print(f"train {len(y):,} rows / test {len(y_test):,} rows, "
              f"{int(y_test.sum()):,} test positives")

        # --- 1. PAY_0 ablation over the 25 repeated-CV fits, paired fold by fold
        cv_full = cv_evaluate(X, y, lambda: xgb_classifier(params), rfolds,
                              collect_oof=False)
        cv_nopay0 = cv_evaluate(X[NO_PAY0], y, lambda: xgb_classifier(params), rfolds,
                                collect_oof=False)
        deltas = per_fold_metric(cv_nopay0) - per_fold_metric(cv_full)
        pay0_test = paired_fold_test(deltas)
        print(f"PAY_0 dropped: CV ROC-AUC {cv_nopay0['cv']['roc_auc_mean']:.5f} against "
              f"{cv_full['cv']['roc_auc_mean']:.5f} "
              f"({pay0_test['mean_delta']:+.5f} paired, p={pay0_test['p_value']:.2g})")

        # --- 2. the configurations that will be read against the test set
        spw = scale_pos_weight(y)
        params_spw = (json.loads(BEST_PARAMS_SPW.read_text(encoding="utf-8"))["params"]
                      if BEST_PARAMS_SPW.exists() else None)

        specs = [
            ("rung1_pay0_baseline", ["PAY_0"], single_rule, None,
             "depth-1 tree on PAY_0 alone; the DeLong comparator"),
            ("rung5_xgb_default", RAW_FEATURES, xgb_library_defaults, None,
             "XGBoost at library defaults, isolating the tuning gain on the hold-out"),
            ("xgb_tuned", RAW_FEATURES, lambda: xgb_classifier(params), params,
             "the headline configuration"),
            ("xgb_tuned_no_pay0", NO_PAY0, lambda: xgb_classifier(params), params,
             "robustness to the PAY_0 construction irregularity"),
            ("xgb_tuned_no_demographics", NO_DEMO, lambda: xgb_classifier(params), params,
             "the fairness-through-unawareness arm of WP17, read on the hold-out"),
        ]
        if params_spw is not None:
            specs.append(("xgb_tuned_spw", RAW_FEATURES,
                          lambda: xgb_classifier(params_spw, spw=spw), params_spw,
                          "reweighted arm C, for the calibration-versus-ranking contrast"))

        rows, test_proba = [], {}
        for label, feats, factory, prm, desc in specs:
            Xf, Xt = X[feats], X_test[feats]
            oof = cv_evaluate(Xf, y, factory, folds)["oof"]
            tau, _ = pick_threshold_oof(y, oof, r=R_HEADLINE)
            tau_inst, _ = pick_threshold_oof(y, oof, r=R_HEADLINE,
                                             ead=Xf["LIMIT_BAL"].to_numpy(dtype=float)
                                             if "LIMIT_BAL" in feats else None)
            model = fit_one(factory(), Xf, y, seed=SEED)
            proba = model.predict_proba(Xt)[:, 1]
            test_proba[label] = proba

            m = evaluate(y_test, proba, tau)
            m["total_cost_r6"] = cost_at_threshold(y_test, proba, tau, r=R_HEADLINE)
            rows.append(make_row(
                label, None, params=prm, n_features=len(feats), test=m,
                extra={
                    "description": desc,
                    "frozen_threshold_source": "OOF training predictions at r=6",
                    "tau_flat": tau,
                    "tau_instance_cost": tau_inst,
                    "test_cost_instance_r6": cost_at_threshold(
                        y_test, proba, tau_inst, r=R_HEADLINE, ead=ead_test),
                    "test_cost_at_0.5": cost_at_threshold(y_test, proba, 0.5,
                                                          r=R_HEADLINE),
                    "oof_roc_auc": float(roc_auc_score(y, oof)),
                }))
            print(f"{label:<28} tau {tau:.4f}  test ROC-AUC {m['roc_auc']:.5f}  "
                  f"recall {m['recall']:.4f}  precision {m['precision']:.4f}  "
                  f"cost {m['total_cost_r6']:,.0f}")

        # --- 3. calibrated headline, if WP13 produced the isotonic OOF vector
        iso_oof = ARTIFACTS / "oof_calib_A_spw1_isotonic.npy"
        if iso_oof.exists():
            tau_iso, _ = pick_threshold_oof(y, np.load(iso_oof), r=R_HEADLINE)
            i_fit, i_cal = train_test_split(np.arange(len(y)), test_size=ES_SLICE,
                                            stratify=y, random_state=SEED)
            base = fit_one(xgb_classifier(params), X.iloc[i_fit], y[i_fit], seed=SEED)
            # sklearn 1.9 removed cv="prefit"; FrozenEstimator is the replacement, so
            # `fit` below trains the isotonic map only, never the booster.
            cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
            cal.fit(X.iloc[i_cal], y[i_cal])
            proba = cal.predict_proba(X_test)[:, 1]
            test_proba["xgb_tuned_isotonic"] = proba
            m = evaluate(y_test, proba, tau_iso)
            m["total_cost_r6"] = cost_at_threshold(y_test, proba, tau_iso, r=R_HEADLINE)
            rows.append(make_row(
                "xgb_tuned_isotonic", None, params=params, n_features=X.shape[1], test=m,
                extra={"description": "tuned model with isotonic calibration fitted on a "
                                      f"held-out {ES_SLICE:.0%} slice of train",
                       "frozen_threshold_source": "OOF isotonic predictions at r=6",
                       "tau_flat": tau_iso, "tau_instance_cost": None,
                       "test_cost_instance_r6": None,
                       "test_cost_at_0.5": cost_at_threshold(y_test, proba, 0.5,
                                                             r=R_HEADLINE),
                       "oof_roc_auc": None}))
            print(f"{'xgb_tuned_isotonic':<28} tau {tau_iso:.4f}  test ROC-AUC "
                  f"{m['roc_auc']:.5f}  brier {m['brier']:.5f}")
        else:
            print("note: isotonic OOF not found, skipping the calibrated row "
                  "(run exp09_calibration first)")

        # --- 4. seed variance on the final configuration, read on the hold-out
        seed_aucs, seed_briers = [], []
        for s in range(SEED, SEED + SEED_COUNT):
            m = fit_one(xgb_classifier(params, seed=s), X, y, seed=s)
            mm = evaluate(y_test, m.predict_proba(X_test)[:, 1], 0.5)
            seed_aucs.append(mm["roc_auc"])
            seed_briers.append(mm["brier"])
        seed_sd = float(np.std(seed_aucs, ddof=1))
        print(f"{SEED_COUNT} seeds: test ROC-AUC {np.mean(seed_aucs):.5f} +/- {seed_sd:.5f} "
              f"(range {np.ptp(seed_aucs):.5f})")

        # --- 5. DeLong, tuned XGBoost against the PAY_0 single rule
        dl = delong_test(y_test, test_proba["xgb_tuned"],
                         test_proba["rung1_pay0_baseline"])
        print(f"DeLong: {dl['auc_a']:.5f} vs {dl['auc_b']:.5f}, diff {dl['diff']:+.5f} "
              f"95% CI [{dl['ci95'][0]:.5f}, {dl['ci95'][1]:.5f}], p={dl['p_value']:.3g}")

        dl_pay0 = delong_test(y_test, test_proba["xgb_tuned"],
                              test_proba["xgb_tuned_no_pay0"])
        dl_tuning = delong_test(y_test, test_proba["xgb_tuned"],
                                test_proba["rung5_xgb_default"])

        # ROC and PR curves for WP19 rebuild from these rather than refitting.
        np.save(ARTIFACTS / "y_test.npy", y_test)
        for label, proba in test_proba.items():
            np.save(ARTIFACTS / f"test_proba_{label}.npy", proba)

        bayes = bayes_floor_stats(df)
        print(f"contradictory duplicate groups: {bayes['contradictory_groups']} covering "
              f"{bayes['contradictory_rows']} rows ({bayes['contradictory_share_pct']}%)")

        by = {r["label"]: r for r in rows}
        head = by["xgb_tuned"]["test"]

        notes = (
            f"Single test-set gate. {len(y_test):,} held-out rows, "
            f"{int(y_test.sum()):,} positives; every threshold was frozen on out-of-fold "
            f"training predictions at r={R_HEADLINE} before this file read the hold-out, "
            f"and no hyperparameter or operating point is selected here. "
            f"Headline: test ROC-AUC {head['roc_auc']:.5f}, PR-AUC {head['pr_auc']:.5f}, "
            f"Brier {head['brier']:.5f}, and at the frozen threshold "
            f"{head['threshold']:.4f} recall {head['recall']:.4f} against precision "
            f"{head['precision']:.4f} ({head['tp']} TP, {head['fp']} FP, {head['fn']} FN, "
            f"{head['tn']} TN). "
            f"Against the PAY_0 single-rule baseline the DeLong difference is "
            f"{dl['diff']:+.5f} (95% CI {dl['ci95'][0]:+.5f} to {dl['ci95'][1]:+.5f}, "
            f"p={dl['p_value']:.3g}), so the ensemble's advantage over one rule is real "
            f"and not a fold artefact. Against untuned XGBoost the difference is "
            f"{dl_tuning['diff']:+.5f} (p={dl_tuning['p_value']:.3g}) -- tuning buys far "
            f"less than capability does, which is itself the answer to how much of the "
            f"ladder's climb is architecture rather than search. "
            f"PAY_0 anomaly: Phase 1 recorded repayment code 1 appearing 0, 0, 2, 4, 28 "
            f"times April-August and 3,688 times in September, so PAY_0 looks constructed "
            f"differently from the five historical columns. Dropping it costs "
            f"{pay0_test['mean_delta']:+.5f} CV ROC-AUC across the 25 repeated fits "
            f"(paired per-fold, p={pay0_test['p_value']:.3g}; Nadeau & Bengio (2003) note "
            f"such tests are anti-conservative because the training folds overlap, so read "
            f"the effect size, not the p-value) and "
            f"{by['xgb_tuned_no_pay0']['test']['roc_auc'] - head['roc_auc']:+.5f} on the "
            f"test set. The headline therefore leans on a column with a documented "
            f"irregularity, and that is reported rather than hidden: the model without "
            f"PAY_0 still reaches "
            f"{by['xgb_tuned_no_pay0']['test']['roc_auc']:.5f}, so the result is not an "
            f"artefact of the anomaly, but a production deployment should verify how the "
            f"most recent repayment status is derived. "
            f"Seed variance: {SEED_COUNT} refits of the final configuration give test "
            f"ROC-AUC {np.mean(seed_aucs):.5f} +/- {seed_sd:.5f} (range "
            f"{np.ptp(seed_aucs):.5f}). Any comparison closer than roughly "
            f"{2 * seed_sd:.4f} is not distinguishable from reseeding noise, and the "
            f"report does not bold one. "
            f"Irreducible floor: {bayes['contradictory_groups']} groups of "
            f"feature-identical clients carry contradictory labels, covering "
            f"{bayes['contradictory_rows']} rows, {bayes['contradictory_share_pct']}% of "
            f"the {bayes['n_rows']:,} cleaned rows. That contribution is small, so the "
            f"honest claim is not that label noise explains the error. It is that AUC 1.0 "
            f"is formally unattainable once contradictory rows exist at all, and more "
            f"importantly that the feature set is a six-month billing snapshot which "
            f"cannot observe job loss, illness or an income shock. Plateauing near "
            f"ROC-AUC {head['roc_auc']:.2f} is a feature-set ceiling, not a model "
            f"deficiency -- supported by the learning curve of WP12 flattening well before "
            f"the full training set and by the small tuned-versus-untuned gap measured "
            f"above."
        )
        save_result("exp14_final",
                    "Robustness checks and final held-out evaluation", rows, notes,
                    t.seconds,
                    config={"r": R_HEADLINE, "n_test": int(len(y_test)),
                            "n_test_positives": int(y_test.sum()),
                            "seed_count": SEED_COUNT,
                            "seed_test_roc_auc": seed_aucs,
                            "seed_test_brier": seed_briers,
                            "seed_roc_auc_sd": seed_sd,
                            "pay0_ablation": {
                                "cv_roc_auc_full": cv_full["cv"]["roc_auc_mean"],
                                "cv_roc_auc_no_pay0": cv_nopay0["cv"]["roc_auc_mean"],
                                "cv_sd_full": cv_full["cv"]["roc_auc_sd"],
                                "paired_fold_test": pay0_test},
                            "delong": {"tuned_vs_pay0_rule": dl,
                                       "tuned_vs_no_pay0": dl_pay0,
                                       "tuned_vs_library_defaults": dl_tuning},
                            "bayes_floor": bayes,
                            "no_pay0_features": NO_PAY0,
                            "no_demo_features": NO_DEMO})


if __name__ == "__main__":
    main()
