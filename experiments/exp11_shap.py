"""WP15 / exp11 -- TreeSHAP attribution. Answers research question 2.

Explanations are computed on held-out rows, because "can post-hoc explanation provide
regulatory-compliant decision auditability" is a question about decisions the model was
not fitted on. SHAP explains the model's output; it does not establish a causal
mechanism, and that distinction is stated in the report.

Attribution is in log-odds (margin) space, where TreeSHAP is exactly additive. The
probability transform is non-additive, so it is used only for the single-client
waterfall.

    python -m experiments.exp11_shap
"""

from __future__ import annotations

import json

import numpy as np
import shap
from scipy import stats

from src.config import ARTIFACTS, SEED
from src.data import BILL_COLS, PAY_COLS, RAW_FEATURES, test_context, training_context
from src.models import xgb_classifier
from src.runner import Timer, fit_one, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"
BACKGROUND_N = 1000
EXPLAIN_N = 2000


def _stratified_sample(X, y, n, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = int(round(n * len(pos) / len(y)))
    take = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n - n_pos, replace=False)])
    take.sort()
    return X.iloc[take], y[take], take


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        tr = training_context(RAW_FEATURES)
        te = test_context(RAW_FEATURES)
        X_tr, y_tr = tr["X"], tr["y"]
        X_te, y_te = te["X_test"], te["y_test"]

        model = fit_one(xgb_classifier(params), X_tr, y_tr, seed=SEED)

        bg, _, _ = _stratified_sample(X_tr, y_tr, BACKGROUND_N, SEED)
        X_exp, y_exp, take = _stratified_sample(X_te, y_te, EXPLAIN_N, SEED)

        # Interventional perturbation breaks the dependence between a feature and its
        # correlates, which is the right choice when the question is "what did the model
        # use", not "what does the data imply".
        ex = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        sv = ex(X_exp, check_additivity=True)
        margin = model.predict(X_exp, output_margin=True)
        add_err = float(np.abs(sv.values.sum(1) + sv.base_values - margin).max())
        assert add_err < 1e-4, f"SHAP additivity violated: {add_err}"
        print(f"additivity max error {add_err:.2e} (log-odds space)")

        # Cross-check against the path-dependent algorithm, which needs no background.
        ex_pd = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv_pd = ex_pd.shap_values(X_exp)

        mean_abs = np.abs(sv.values).mean(0)
        mean_abs_pd = np.abs(sv_pd).mean(0)
        agree_rho = float(stats.spearmanr(mean_abs, mean_abs_pd).statistic)
        print(f"interventional vs path-dependent mean|SHAP| Spearman rho {agree_rho:.4f}")

        np.save(ARTIFACTS / "shap_values_test.npy", sv.values)
        np.save(ARTIFACTS / "shap_base_values_test.npy", np.asarray(sv.base_values))
        np.save(ARTIFACTS / "shap_explain_idx.npy", take)
        X_exp.to_csv(ARTIFACTS / "shap_explain_X.csv", index=False)

        order = np.argsort(-mean_abs)
        rows = []
        for rank, j in enumerate(order, start=1):
            f = RAW_FEATURES[j]
            rows.append(make_row(
                f, None, n_features=1,
                extra={
                    "rank": rank,
                    "mean_abs_shap": float(mean_abs[j]),
                    "mean_shap": float(sv.values[:, j].mean()),
                    "mean_abs_shap_path_dependent": float(mean_abs_pd[j]),
                    "share_of_total": float(mean_abs[j] / mean_abs.sum()),
                    "group": ("pay_status" if f in PAY_COLS else
                              "bill" if f in BILL_COLS else "other"),
                }))
        print("top 8 by mean|SHAP|:",
              ", ".join(f"{RAW_FEATURES[j]} {mean_abs[j]:.4f}" for j in order[:8]))

        # Convergent validation: does the model's internal response to PAY_0 reproduce
        # the EDA default-rate gradient?
        pay0 = X_exp["PAY_0"].to_numpy()
        j0 = RAW_FEATURES.index("PAY_0")
        pay0_profile = {}
        for code in sorted(set(pay0.tolist())):
            m = pay0 == code
            pay0_profile[int(code)] = {
                "n": int(m.sum()),
                "mean_shap_logodds": float(sv.values[m, j0].mean()),
                "observed_default_rate": float(y_exp[m].mean()),
            }
        codes = sorted(pay0_profile)
        rho_shap_rate = float(stats.spearmanr(
            [pay0_profile[c]["mean_shap_logodds"] for c in codes],
            [pay0_profile[c]["observed_default_rate"] for c in codes]).statistic)

        # Waterfall pair: the most confident true positive and the worst false negative.
        proba = model.predict_proba(X_exp)[:, 1]
        tp_idx = int(np.argmax(np.where(y_exp == 1, proba, -np.inf)))
        fn_idx = int(np.argmin(np.where(y_exp == 1, proba, np.inf)))
        cases = {
            "true_positive": {"row": tp_idx, "proba": float(proba[tp_idx]),
                              "y": int(y_exp[tp_idx])},
            "false_negative": {"row": fn_idx, "proba": float(proba[fn_idx]),
                               "y": int(y_exp[fn_idx])},
        }
        print(f"waterfall cases: TP p={proba[tp_idx]:.4f}, FN p={proba[fn_idx]:.4f}")

        top = [RAW_FEATURES[j] for j in order[:5]]
        bill_share = float(sum(mean_abs[RAW_FEATURES.index(c)] for c in BILL_COLS)
                           / mean_abs.sum())
        pay_share = float(sum(mean_abs[RAW_FEATURES.index(c)] for c in PAY_COLS)
                          / mean_abs.sum())

        notes = (
            f"TreeSHAP on {len(X_exp):,} stratified held-out rows with a {BACKGROUND_N}-row "
            f"stratified background from train. Additivity holds to {add_err:.1e} in "
            f"log-odds space. Interventional and path-dependent mean|SHAP| rankings agree "
            f"at Spearman rho {agree_rho:.4f}. "
            f"Top five by mean|SHAP|: {', '.join(top)}. Repayment-status columns carry "
            f"{pay_share:.1%} of total attribution against {bill_share:.1%} for the six "
            f"BILL_AMT columns. "
            f"Convergent validation: mean SHAP by PAY_0 code correlates with the observed "
            f"default rate at Spearman rho {rho_shap_rate:.4f}, so the model's internals "
            f"independently reproduce the non-monotonic gradient Phase 1 measured in the "
            f"raw data (12.8% at -1, 16.8% at 0, 34.0% at 1, 69.2% at 2). The model was "
            f"never told PAY_n is ordinal; it recovered the shape from the data. "
            f"Caveat on collinearity: BILL_AMT1-6 carry VIF 13.9-25.7 (Phase 1 Fig 2.14), "
            f"so TreeSHAP splits credit arbitrarily among them. Individual BILL_AMT "
            f"attributions are unstable and must be read as a group total, not per column. "
            f"Caveat on interpretation: SHAP attributes the model's output, not a causal "
            f"mechanism -- it answers 'why did this model decide that', which is the "
            f"auditability question, and not 'what would happen if we changed this'."
        )
        save_result("exp11_shap", "TreeSHAP attribution on held-out decisions",
                    rows, notes, t.seconds,
                    config={"background_n": BACKGROUND_N, "explain_n": len(X_exp),
                            "feature_perturbation": "interventional",
                            "space": "log-odds (margin)",
                            "additivity_max_error": add_err,
                            "path_dependent_rank_rho": agree_rho,
                            "pay0_profile": pay0_profile,
                            "pay0_shap_vs_rate_rho": rho_shap_rate,
                            "waterfall_cases": cases,
                            "group_share": {"pay_status": pay_share, "bill": bill_share}})


if __name__ == "__main__":
    main()
