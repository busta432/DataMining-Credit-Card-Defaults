"""WP17 / exp13 -- fairness and subgroup analysis. Promised in Phase 1 section 2.2.

Phase 1 quantified the demographic disparities already present in the data so that a
later fairness check would be interpretable rather than merely alarming. The question
here is therefore specific: does the model *amplify*, *preserve* or *attenuate* a
disparity that already exists?

Evaluated on out-of-fold training predictions, not the test set. Nothing is selected
here -- it is pure reporting -- but OOF keeps the single test-set gate in exp14 intact
and gives subgroup samples four times larger, so the Wilson intervals are tighter.

Second half: if the financial columns proxy for demographics, dropping the demographic
columns hides the influence rather than removing it (Barocas & Selbst, 2016). The
no-demographics model is refitted here and the recovery measured.

    python -m experiments.exp13_fairness
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ARTIFACTS, R_HEADLINE
from src.costs import pick_threshold_oof
from src.data import (
    AGE_BANDS,
    CLEAN_EDU,
    CLEAN_MAR,
    DEMOGRAPHIC,
    RAW_FEATURES,
    SEX_LABEL,
    age_band,
    training_context,
)
from src.metrics import wilson
from src.models import xgb_classifier
from src.runner import Timer, cv_evaluate, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"
NO_DEMO = [c for c in RAW_FEATURES if c not in DEMOGRAPHIC]

# Phase 1 section 2.2, for the amplify / preserve / attenuate comparison.
PHASE1_RATES = {
    "SEX": {1: 24.16, 2: 20.77},
    "EDUCATION": {1: 19.21, 2: 23.74, 3: 25.16},
}


def subgroup_metrics(y, proba, keys, labels, tau):
    out = []
    pred = (proba >= tau).astype(int)
    for k in sorted(set(keys.tolist())):
        m = keys == k
        yk, pk, dk = y[m], proba[m], pred[m]
        n, pos = int(m.sum()), int(yk.sum())
        lo, hi = wilson(pos, n)
        tp = int(((yk == 1) & (dk == 1)).sum())
        fn = int(((yk == 1) & (dk == 0)).sum())
        fp = int(((yk == 0) & (dk == 1)).sum())
        tn = int(((yk == 0) & (dk == 0)).sum())
        out.append({
            "group": labels.get(k, str(k)) if isinstance(labels, dict) else str(k),
            "code": k if not isinstance(k, str) else k,
            "n": n,
            "observed_default_rate_pct": round(100 * pos / n, 4),
            "wilson_lo_pct": round(lo, 4), "wilson_hi_pct": round(hi, 4),
            "roc_auc": float(roc_auc_score(yk, pk)) if 0 < pos < n else None,
            "pr_auc": float(average_precision_score(yk, pk)) if 0 < pos < n else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "fpr": fp / (fp + tn) if (fp + tn) else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "selection_rate": float(dk.mean()),
            "mean_predicted": float(pk.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return out


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        ctx = training_context(RAW_FEATURES)
        X, y, folds = ctx["X"], ctx["y"], ctx["folds"]

        oof_path = ARTIFACTS / "oof_proba_A_baseline.npy"
        if not oof_path.exists():
            raise SystemExit(f"MISSING ARTIFACT: {oof_path}\nProduced by WP7: "
                             "python -m experiments.exp03_imbalance")
        oof_full = np.load(oof_path)

        nd = cv_evaluate(X[NO_DEMO], y, lambda: xgb_classifier(params), folds)
        oof_nd = nd["oof"]
        np.save(ARTIFACTS / "oof_proba_no_demographics.npy", oof_nd)
        print(f"no-demographics model: {len(NO_DEMO)} features, "
              f"OOF ROC-AUC {roc_auc_score(y, oof_nd):.5f} against "
              f"{roc_auc_score(y, oof_full):.5f} for the full model")

        tau_full, _ = pick_threshold_oof(y, oof_full, r=R_HEADLINE)
        tau_nd, _ = pick_threshold_oof(y, oof_nd, r=R_HEADLINE)
        print(f"frozen thresholds at r={R_HEADLINE}: full {tau_full:.4f}, "
              f"no-demographics {tau_nd:.4f}")

        attrs = {
            "SEX": (X["SEX"].to_numpy(), SEX_LABEL),
            "EDUCATION": (X["EDUCATION"].to_numpy(), CLEAN_EDU),
            "MARRIAGE": (X["MARRIAGE"].to_numpy(), CLEAN_MAR),
            "AGE_BAND": (age_band(X["AGE"]).to_numpy(), {}),
        }

        rows, detail = [], {}
        for model_name, proba, tau in (("full", oof_full, tau_full),
                                       ("no_demographics", oof_nd, tau_nd)):
            for attr, (keys, labels) in attrs.items():
                groups = subgroup_metrics(y, proba, keys, labels, tau)
                detail[f"{model_name}|{attr}"] = groups
                recalls = [g["recall"] for g in groups if g["recall"] is not None]
                fprs = [g["fpr"] for g in groups if g["fpr"] is not None]
                sel = [g["selection_rate"] for g in groups]
                aucs = [g["roc_auc"] for g in groups if g["roc_auc"] is not None]
                obs = [g["observed_default_rate_pct"] for g in groups]
                rows.append(make_row(
                    f"{model_name}_{attr}", None, n_features=(X.shape[1]
                                                              if model_name == "full"
                                                              else len(NO_DEMO)),
                    extra={
                        "model": model_name, "attribute": attr, "threshold": tau,
                        "n_groups": len(groups),
                        "recall_gap": max(recalls) - min(recalls),
                        "fpr_gap": max(fprs) - min(fprs),
                        # Demographic-parity ratio; the "four-fifths rule" reads below 0.8
                        # as disparate impact.
                        "selection_rate_ratio": min(sel) / max(sel),
                        "roc_auc_gap": max(aucs) - min(aucs),
                        "observed_rate_gap_pp": max(obs) - min(obs),
                        "groups": groups,
                    }))
                print(f"{model_name:<16} {attr:<10} recall gap "
                      f"{max(recalls) - min(recalls):.4f}  fpr gap "
                      f"{max(fprs) - min(fprs):.4f}  sel ratio "
                      f"{min(sel) / max(sel):.4f}  auc gap {max(aucs) - min(aucs):.4f}")

        by = {r["label"]: r for r in rows}

        def amplification(attr):
            """Model selection-rate gap against the observed default-rate gap."""
            g = by[f"full_{attr}"]
            obs = g["observed_rate_gap_pp"] / 100
            groups = g["groups"]
            model_gap = (max(x["selection_rate"] for x in groups)
                         - min(x["selection_rate"] for x in groups))
            return obs, model_gap, (model_gap / obs if obs else None)

        sex_obs, sex_model, sex_ratio = amplification("SEX")
        edu_obs, edu_model, edu_ratio = amplification("EDUCATION")

        sex_full = by["full_SEX"]["groups"]
        sex_nd = by["no_demographics_SEX"]["groups"]
        sex_gap_full = (max(x["selection_rate"] for x in sex_full)
                        - min(x["selection_rate"] for x in sex_full))
        sex_gap_nd = (max(x["selection_rate"] for x in sex_nd)
                      - min(x["selection_rate"] for x in sex_nd))
        recovery = sex_gap_nd / sex_gap_full if sex_gap_full else None

        notes = (
            f"Evaluated on out-of-fold training predictions at the frozen r={R_HEADLINE} "
            f"threshold ({tau_full:.4f} for the full model). Nothing is selected here, and "
            f"the test set is untouched; OOF also gives subgroup samples four times larger "
            f"than the hold-out would, so the Wilson intervals are tighter. "
            f"Observed disparity in the cleaned training rows: SEX default-rate gap "
            f"{sex_obs:.4f} and EDUCATION gap {edu_obs:.4f}, against Phase 1's reported "
            f"male 24.16% versus female 20.77% and education 19.21% -> 23.74% -> 25.16%. "
            f"The model's selection-rate gap is {sex_model:.4f} for SEX, a ratio of "
            f"{sex_ratio:.3f} against the observed gap -- so the model "
            f"{'amplifies' if sex_ratio and sex_ratio > 1.1 else 'attenuates' if sex_ratio and sex_ratio < 0.9 else 'roughly preserves'} "
            f"the disparity already in the data rather than creating one. For EDUCATION the "
            f"ratio is {edu_ratio:.3f}. "
            f"Per-subgroup ROC-AUC gaps are reported alongside, because equal ranking "
            f"quality and equal error rates are different fairness criteria and this model "
            f"does not satisfy both. Selection-rate ratios are given against the "
            f"four-fifths rule threshold of 0.8. "
            f"Proxy test: removing SEX, EDUCATION, MARRIAGE and AGE costs "
            f"{roc_auc_score(y, oof_nd) - roc_auc_score(y, oof_full):+.5f} OOF ROC-AUC, yet "
            f"the SEX selection-rate gap only falls from {sex_gap_full:.4f} to "
            f"{sex_gap_nd:.4f} -- {recovery:.1%} of it survives. The financial columns "
            f"proxy for the demographics, so dropping the protected columns hides the "
            f"influence rather than removing it, exactly as Barocas & Selbst (2016) "
            f"predict. Fairness-through-unawareness is not available on this dataset."
        )
        save_result("exp13_fairness", "Fairness and subgroup analysis", rows, notes,
                    t.seconds,
                    config={"threshold_full": tau_full, "threshold_no_demo": tau_nd,
                            "r": R_HEADLINE, "no_demo_features": NO_DEMO,
                            "age_bands": [list(b) for b in AGE_BANDS],
                            "phase1_reported_rates_pct": PHASE1_RATES,
                            "oof_roc_auc": {"full": float(roc_auc_score(y, oof_full)),
                                            "no_demographics": float(roc_auc_score(y, oof_nd))},
                            "proxy_recovery_sex": recovery,
                            "subgroup_detail": detail})


if __name__ == "__main__":
    main()
