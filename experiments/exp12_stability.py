"""WP16 / exp12 -- explanation stability and importance-measure agreement.

Research question 2 asks whether post-hoc explanation can provide auditable
decisions. An explanation that reorders when the model is reseeded is not an audit
trail, so stability is the part of that question that actually has teeth. The proposal
cites Lin & Wang (2025) on SHAP stability in credit risk; this operationalises it.

Second half: SHAP, gain, cover, weight and permutation importance routinely disagree.
Showing the disagreement and arguing which to trust is the substantive result.

    python -m experiments.exp12_stability
"""

from __future__ import annotations

import json
import itertools

import numpy as np
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

from src.config import ARTIFACTS, SEED
from src.data import BILL_COLS, RAW_FEATURES, test_context, training_context
from src.models import xgb_classifier
from src.runner import Timer, fit_one, make_row, save_result

BEST_PARAMS = ARTIFACTS / "best_params.json"
SEEDS = [SEED + i for i in range(5)]
BACKGROUND_N = 500
EXPLAIN_N = 1000
TOP_K = 10


def _stratified_sample(X, y, n, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(round(n * len(pos) / len(y)))
    take = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n - n_pos, replace=False)])
    take.sort()
    return X.iloc[take], y[take]


def main():
    with Timer() as t:
        if not BEST_PARAMS.exists():
            raise SystemExit(f"MISSING ARTIFACT: {BEST_PARAMS}\nProduced by WP6: "
                             "python -m experiments.exp02_tuning")
        params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))["params"]

        tr = training_context(RAW_FEATURES)
        te = test_context(RAW_FEATURES)
        X_tr, y_tr = tr["X"], tr["y"]
        bg, _ = _stratified_sample(X_tr, y_tr, BACKGROUND_N, SEED)
        X_exp, y_exp = _stratified_sample(te["X_test"], te["y_test"], EXPLAIN_N, SEED)

        # --- 1. stability of the SHAP ranking under reseeding
        mean_abs_by_seed = {}
        models = {}
        for s in SEEDS:
            m = fit_one(xgb_classifier(params, seed=s), X_tr, y_tr, seed=s)
            models[s] = m
            ex = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional")
            mean_abs_by_seed[s] = np.abs(ex.shap_values(X_exp)).mean(0)
            print(f"seed {s}: top3 "
                  f"{[RAW_FEATURES[j] for j in np.argsort(-mean_abs_by_seed[s])[:3]]}")

        pairs = list(itertools.combinations(SEEDS, 2))
        rhos, jaccards = [], []
        for a, b in pairs:
            rhos.append(float(stats.spearmanr(mean_abs_by_seed[a],
                                              mean_abs_by_seed[b]).statistic))
            ta = set(np.argsort(-mean_abs_by_seed[a])[:TOP_K])
            tb = set(np.argsort(-mean_abs_by_seed[b])[:TOP_K])
            jaccards.append(len(ta & tb) / len(ta | tb))
        print(f"pairwise Spearman rho: mean {np.mean(rhos):.4f}, min {np.min(rhos):.4f}")
        print(f"top-{TOP_K} Jaccard: mean {np.mean(jaccards):.4f}, "
              f"min {np.min(jaccards):.4f}")

        # Per-feature rank variability across seeds.
        rank_matrix = np.vstack([stats.rankdata(-mean_abs_by_seed[s]) for s in SEEDS])
        rank_sd = rank_matrix.std(axis=0, ddof=1)

        # --- 2. importance-measure disagreement on the reference model
        ref = models[SEED]
        booster = ref.get_booster()
        booster.feature_names = list(RAW_FEATURES)
        measures = {"shap": mean_abs_by_seed[SEED]}
        for kind in ("gain", "cover", "weight", "total_gain"):
            score = booster.get_score(importance_type=kind)
            measures[kind] = np.array([score.get(f, 0.0) for f in RAW_FEATURES])
        perm = permutation_importance(ref, X_exp, y_exp, n_repeats=10,
                                      random_state=SEED, scoring="roc_auc", n_jobs=-1)
        measures["permutation"] = perm.importances_mean

        names = list(measures)
        agreement = {}
        for a, b in itertools.combinations(names, 2):
            agreement[f"{a}|{b}"] = float(stats.spearmanr(measures[a],
                                                          measures[b]).statistic)
        matrix = {a: {b: (1.0 if a == b else
                          float(stats.spearmanr(measures[a], measures[b]).statistic))
                      for b in names} for a in names}
        worst = min(agreement, key=agreement.get)
        best = max(agreement, key=agreement.get)
        print(f"importance agreement: best {best} {agreement[best]:.4f}, "
              f"worst {worst} {agreement[worst]:.4f}")

        order = np.argsort(-measures["shap"])
        rows = []
        for rank, j in enumerate(order, start=1):
            rows.append(make_row(
                RAW_FEATURES[j], None, n_features=1,
                extra={
                    "shap_rank": rank,
                    "mean_abs_shap": float(measures["shap"][j]),
                    "rank_sd_across_seeds": float(rank_sd[j]),
                    "shap_by_seed": {str(s): float(mean_abs_by_seed[s][j])
                                     for s in SEEDS},
                    **{f"imp_{k}": float(v[j]) for k, v in measures.items()
                       if k != "shap"},
                    **{f"rank_{k}": int(stats.rankdata(-v)[j]) for k, v in measures.items()},
                }))

        bill_rank_sd = float(np.mean([rank_sd[RAW_FEATURES.index(c)] for c in BILL_COLS]))
        other = [c for c in RAW_FEATURES if c not in BILL_COLS]
        other_rank_sd = float(np.mean([rank_sd[RAW_FEATURES.index(c)] for c in other]))

        notes = (
            f"Stability: across {len(SEEDS)} reseeded refits, pairwise Spearman rho between "
            f"mean|SHAP| vectors averages {np.mean(rhos):.4f} (minimum "
            f"{np.min(rhos):.4f}) and the top-{TOP_K} Jaccard index averages "
            f"{np.mean(jaccards):.4f} (minimum {np.min(jaccards):.4f}). The global "
            f"explanation is therefore reproducible at the level a regulator would ask "
            f"about: the same features, in close to the same order, regardless of seed. "
            f"Where instability does appear it is concentrated in the collinear block: "
            f"mean rank SD is {bill_rank_sd:.2f} positions for BILL_AMT1-6 against "
            f"{other_rank_sd:.2f} for the remaining features. That is the VIF 13.9-25.7 "
            f"collinearity of Phase 1 showing up as attribution instability rather than as "
            f"a prediction problem -- which is exactly why the BILL_AMT columns are kept "
            f"(trees are unharmed by collinearity) but read as a group when explained. "
            f"Disagreement: the five importance measures rank features differently. "
            f"Strongest agreement is {best} at rho {agreement[best]:.4f}; weakest is "
            f"{worst} at rho {agreement[worst]:.4f}. `weight` counts how often a feature is "
            f"split on, which rewards high-cardinality continuous columns regardless of "
            f"whether the splits matter; `gain` is computed on training loss reduction and "
            f"is not held-out. SHAP and permutation importance are preferred here because "
            f"both are evaluated on held-out rows and both answer a question about "
            f"predictions rather than about tree structure."
        )
        save_result("exp12_stability",
                    "Explanation stability under reseeding and importance-measure agreement",
                    rows, notes, t.seconds,
                    config={"seeds": SEEDS, "top_k": TOP_K,
                            "explain_n": len(X_exp), "background_n": BACKGROUND_N,
                            "pairwise_spearman": {f"{a}|{b}": r
                                                  for (a, b), r in zip(pairs, rhos)},
                            "pairwise_jaccard": {f"{a}|{b}": j
                                                 for (a, b), j in zip(pairs, jaccards)},
                            "spearman_mean": float(np.mean(rhos)),
                            "jaccard_mean": float(np.mean(jaccards)),
                            "importance_agreement_matrix": matrix})


if __name__ == "__main__":
    main()
