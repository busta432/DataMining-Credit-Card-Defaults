"""Assemble guide/index.html from guide/parts/*.html with live data from results/.

The guide is a single self-contained file so it opens over file:// with no server:
browsers block fetch() against local JSON, so the payload is inlined instead.

Every number rendered in the guide comes from results/*.json through PAYLOAD below --
the same rule the report follows. Nothing is retyped by hand.

    python guide/build_guide.py
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARTS = ROOT / "guide" / "parts"
OUT = ROOT / "guide" / "index.html"


def load(exp):
    return json.loads((RESULTS / f"{exp}.json").read_text(encoding="utf-8"))


def rows_of(exp, keys=("label",), extra=()):
    """Flatten rows[] to the fields the guide actually plots."""
    d = load(exp)
    out = []
    for r in d["rows"]:
        cv = r.get("cv") or {}
        rec = {
            "label": r["label"],
            "n_features": r.get("n_features"),
            "auc": cv.get("roc_auc_mean"),
            "sd": cv.get("roc_auc_sd"),
            "pr": cv.get("pr_auc_mean"),
            "brier": cv.get("brier_mean"),
        }
        for k in extra:
            rec[k] = r.get(k)
        out.append(rec)
    return out


def optuna_trials():
    """Per-trial objective and parameters for the tuning-surface widget.

    This is the one payload block that does NOT come from results/. exp02 serialises
    the best trial and the aggregate counts but not the 150-row history, and the
    history is what makes "the objective surface is flat" visible rather than merely
    asserted. The study database is gitignored (regenerable, and large), so the block
    is optional: on a fresh checkout without it the widget renders an explanatory note
    instead of silently showing nothing. Read straight from SQLite rather than through
    optuna so the guide does not import a 40 MB dependency to draw a scatter plot.
    """
    db = ROOT / "artifacts" / "optuna_study.db"
    if not db.exists():
        return None

    import sqlite3

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT t.trial_id, t.number, t.state, v.value FROM trials t "
            "LEFT JOIN trial_values v ON v.trial_id = t.trial_id ORDER BY t.number"
        ).fetchall()
        params: dict[int, dict] = {}
        for tid, name, val in con.execute(
            "SELECT trial_id, param_name, param_value FROM trial_params"
        ):
            params.setdefault(tid, {})[name] = val
    finally:
        con.close()

    trials = [
        {"n": num, "state": state, "value": value, "params": params.get(tid, {})}
        for tid, num, state, value in rows
    ]
    done = [t["value"] for t in trials if t["state"] == "COMPLETE" and t["value"]]
    return {
        "trials": trials,
        "lo": min(done),
        "hi": max(done),
        "median": sorted(done)[len(done) // 2],
        "source": "artifacts/optuna_study.db",
    }


def recover_confusion(cost):
    """Recover the out-of-fold confusion matrix at all 51 swept thresholds.

    exp10 serialises only (grid, total_cost) per cost ratio, but for the flat scheme
    cost(r) = r*FN + FP exactly, so two ratios give two equations in two unknowns:

        FN = cost(r=2) - cost(r=1)        FP = 2*cost(r=1) - cost(r=2)

    The remaining four ratios then over-determine the system, which is why they are
    asserted rather than ignored -- if the recovery were wrong, r=20 would not match
    to the last unit. This is reconstruction from committed results, not a re-run:
    the widget's confusion matrix is the same out-of-fold one the report quotes.
    """
    curves = cost["config"]["cost_curves"]
    grid = curves["r1_flat"]["grid"]
    c1, c2 = curves["r1_flat"]["cost"], curves["r2_flat"]["cost"]
    fn = [b - a for a, b in zip(c1, c2)]
    fp = [2 * a - b for a, b in zip(c1, c2)]

    for r in (5, 6, 10, 20):
        want = curves[f"r{r}_flat"]["cost"]
        got = [r * a + b for a, b in zip(fn, fp)]
        if max(abs(x - y) for x, y in zip(got, want)) > 1e-6:
            raise SystemExit(f"confusion recovery failed to reproduce r={r}")

    # tau -> 0 accepts nobody as negative, so FP is the whole negative class;
    # tau -> 1 flags nobody, so FN is the whole positive class.
    pos, neg = int(fn[-1]), int(fp[0])
    return {
        "grid": grid,
        "fn": [int(x) for x in fn],
        "fp": [int(x) for x in fp],
        "pos": pos,
        "neg": neg,
        "n": pos + neg,
    }


def build_payload():
    ladder = load("exp01_ladder")
    tuning = load("exp02_tuning")
    imb = load("exp03_imbalance")
    calib = load("exp09_calibration")
    cost = load("exp10_cost")
    shap = load("exp11_shap")
    stab = load("exp12_stability")
    fair = load("exp13_fairness")
    final = load("exp14_final")

    # The r=6 flat cost curve drives the interactive threshold explorer. It is the
    # real out-of-fold sweep, so the widget reproduces the reported optimum exactly.
    r6 = cost["config"]["cost_curves"]["r6_flat"]
    oof = recover_confusion(cost)

    return {
        "meta": {
            "prevalence": ladder["config"]["prevalence"],
            "versions": final["versions"],
            "runtimes": {
                e: load(e)["runtime_sec"]
                for e in sorted(p.stem for p in RESULTS.glob("exp*.json"))
            },
        },
        "ladder": {
            "rows": rows_of(
                "exp01_ladder",
                extra=("model", "interpretability", "interp_cost",
                       "accuracy_at_0.5", "recall_at_0.5"),
            ),
            "notes": ladder["notes"],
        },
        "tuning": {
            "best_params": tuning["rows"][0]["params"],
            "space": tuning["config"]["search_space"],
            "gain": tuning["config"]["tuning_gain_roc_auc"],
            "pooled_sd": tuning["config"]["pooled_fold_sd"],
            "n_complete": tuning["config"]["n_complete"],
            "n_pruned": tuning["config"]["n_pruned"],
            "n_trees": tuning["rows"][0].get("n_trees"),
            "default_auc": (tuning["rows"][1]["cv"] or {})["roc_auc_mean"],
            "tuned_auc": (tuning["rows"][0]["cv"] or {})["roc_auc_mean"],
            "n_trials": tuning["config"]["n_trials"],
            "runtime": tuning["runtime_sec"],
            "study": optuna_trials(),
            "notes": tuning["notes"],
        },
        "imbalance": {
            "rows": rows_of(
                "exp03_imbalance",
                extra=("oof_threshold", "oof_recall", "oof_precision",
                       "oof_cost_r6", "spw", "sampler"),
            ),
            "spw": imb["config"]["scale_pos_weight"],
            "notes": imb["notes"],
        },
        "calibration": {
            "rows": [
                {
                    "label": r["label"], "arm": r.get("arm"), "method": r.get("method"),
                    "ece": r.get("ece"), "mean_predicted": r.get("mean_predicted"),
                    "brier": (r.get("cv") or {}).get("brier_mean"),
                    "auc": (r.get("cv") or {}).get("roc_auc_mean"),
                }
                for r in calib["rows"]
            ],
            "curves": calib["config"]["reliability_curves"],
            "prevalence": calib["config"]["prevalence"],
            "notes": calib["notes"],
        },
        "oof": oof,
        "cost": {
            "grid": r6["grid"],
            "curve": r6["cost"],
            "elkan": cost["config"]["elkan"],
            "mae": cost["config"]["mae_vs_elkan"],
            "sweep": [
                {
                    "label": r["label"], "r": r.get("r"), "scheme": r.get("scheme"),
                    "tau": r.get("tau_empirical"), "p_star": r.get("elkan_p_star"),
                    "gap": r.get("tau_minus_p_star"), "min_cost": r.get("min_cost"),
                    "cost_at_half": r.get("cost_at_0.5"),
                    "saving": r.get("saving_vs_0.5"),
                    "recall": r.get("recall"), "precision": r.get("precision"),
                }
                for r in cost["rows"]
            ],
            "notes": cost["notes"],
        },
        "shap": {
            "top": [
                {
                    "feature": r["label"], "rank": r.get("rank"),
                    "mean_abs": r.get("mean_abs_shap"),
                    "mean_signed": r.get("mean_shap"),
                    "path_dep": r.get("mean_abs_shap_path_dependent"),
                    "share": r.get("share_of_total"), "group": r.get("group"),
                }
                for r in shap["rows"]
            ],
            "pay0": shap["config"]["pay0_profile"],
            "additivity": shap["config"]["additivity_max_error"],
            "path_rho": shap["config"]["path_dependent_rank_rho"],
            "pay0_rho": shap["config"]["pay0_shap_vs_rate_rho"],
            "background_n": shap["config"]["background_n"],
            "explain_n": shap["config"]["explain_n"],
            "waterfall": shap["config"]["waterfall_cases"],
            "group_share": shap["config"]["group_share"],
            "notes": shap["notes"],
        },
        "stability": {
            "spearman_mean": stab["config"]["spearman_mean"],
            "jaccard_mean": stab["config"]["jaccard_mean"],
            "pairwise_spearman": stab["config"]["pairwise_spearman"],
            "pairwise_jaccard": stab["config"]["pairwise_jaccard"],
            "agreement": stab["config"]["importance_agreement_matrix"],
            "seeds": stab["config"]["seeds"],
            "top_k": stab["config"]["top_k"],
            "rows": [
                {
                    "feature": r["label"], "shap_rank": r.get("shap_rank"),
                    "mean_abs": r.get("mean_abs_shap"),
                    "rank_sd": r.get("rank_sd_across_seeds"),
                    "by_seed": r.get("shap_by_seed"),
                    "gain": r.get("imp_gain"), "cover": r.get("imp_cover"),
                    "weight": r.get("imp_weight"),
                    "total_gain": r.get("imp_total_gain"),
                    "permutation": r.get("imp_permutation"),
                    "ranks": {
                        m: r.get(f"rank_{m}") for m in
                        ("shap", "gain", "cover", "weight", "total_gain", "permutation")
                    },
                }
                for r in stab["rows"]
            ],
            "notes": stab["notes"],
        },
        "fairness": {
            "rows": [
                {
                    "label": r["label"], "model": r.get("model"),
                    "attribute": r.get("attribute"),
                    "recall_gap": r.get("recall_gap"), "fpr_gap": r.get("fpr_gap"),
                    "sel_ratio": r.get("selection_rate_ratio"),
                    "auc_gap": r.get("roc_auc_gap"),
                    "observed_gap_pp": r.get("observed_rate_gap_pp"),
                    "groups": r.get("groups"),
                }
                for r in fair["rows"]
            ],
            "proxy_recovery_sex": fair["config"]["proxy_recovery_sex"],
            "oof_auc": fair["config"]["oof_roc_auc"],
            "threshold": fair["config"]["threshold_full"],
            "phase1": fair["config"]["phase1_reported_rates_pct"],
            "n_no_demo": len(fair["config"]["no_demo_features"]),
            "notes": fair["notes"],
        },
        "final": {
            "rows": [
                {"label": r["label"], "n_features": r.get("n_features"),
                 **{k: (r.get("test") or {}).get(k) for k in
                    ("roc_auc", "pr_auc", "brier", "threshold", "recall",
                     "precision", "tp", "fp", "fn", "tn", "total_cost_r6")}}
                for r in final["rows"]
            ],
            "delong": final["config"]["delong"],
            "seeds": final["config"]["seed_test_roc_auc"],
            "seed_brier": final["config"]["seed_test_brier"],
            "seed_sd": final["config"]["seed_roc_auc_sd"],
            "pay0_ablation": final["config"]["pay0_ablation"],
            "bayes_floor": final["config"]["bayes_floor"],
            "n_test": final["config"]["n_test"],
            "n_test_pos": final["config"]["n_test_positives"],
            "notes": final["notes"],
        },
        "ablations": {
            "encoding": {
                "rows": rows_of("exp04_encoding", extra=("description",)),
                "spread": load("exp04_encoding")["config"]["auc_spread"],
                "notes": load("exp04_encoding")["notes"],
            },
            "engineered": {
                "rows": rows_of("exp05_engineered", extra=("n_engineered", "features")),
                "names": load("exp05_engineered")["config"]["engineered"],
                "rfe": load("exp05_engineered")["config"]["rfe_selected"],
                "notes": load("exp05_engineered")["notes"],
            },
            "groups": {
                "rows": rows_of("exp06_groups", extra=("description",)),
                "loo": load("exp06_groups")["config"]["leave_one_out_loss"],
                "only": load("exp06_groups")["config"]["only_one_auc"],
                "defs": load("exp06_groups")["config"]["groups"],
                "notes": load("exp06_groups")["notes"],
            },
            "recency": {
                "rows": rows_of("exp07_recency", extra=("k_months", "months")),
                "marginal": load("exp07_recency")["config"]["marginal_gain_per_month"],
                "smallest_k": load("exp07_recency")["config"]["smallest_k_within_one_sd"],
                "notes": load("exp07_recency")["notes"],
            },
            "learning": {
                "rows": rows_of(
                    "exp08_learning_curve",
                    extra=("fraction", "n_train_rows_mean", "train_roc_auc_mean",
                           "train_roc_auc_sd", "gap_train_minus_val"),
                ),
                "notes": load("exp08_learning_curve")["notes"],
            },
        },
    }


def main():
    parts = sorted(PARTS.glob("*.html"))
    if not parts:
        raise SystemExit(f"no parts found in {PARTS}")
    html = "\n".join(p.read_text(encoding="utf-8") for p in parts)

    payload = json.dumps(build_payload(), separators=(",", ":"))
    # </script> inside a JSON string would close the host <script> tag early.
    payload = payload.replace("</", "<\\/")

    # The marker carries a `null` fallback so the part file is valid JavaScript on
    # its own; the payload replaces the marker AND that fallback.
    marker = "/*__DATA__*/null"
    if marker not in html:
        raise SystemExit(f"marker {marker} not found in the assembled parts")
    html = html.replace(marker, payload)

    # Sanity: no leftover build markers.
    stray = re.findall(r"/\*__[A-Z_]+__\*/", html)
    if stray:
        raise SystemExit(f"unreplaced markers: {set(stray)}")

    OUT.write_text(html, encoding="utf-8")
    kb = len(html.encode("utf-8")) / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB from {len(parts)} parts, "
          f"{len(payload) / 1024:.0f} KB data)")


if __name__ == "__main__":
    main()
