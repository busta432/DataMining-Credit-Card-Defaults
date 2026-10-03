"""WP19 -- every Phase 2 figure, rebuilt from results/ and artifacts/ alone.

No figure refits a model. That is the point: the report's figures and its tables read the
same serialised numbers, so they cannot drift apart, and a figure can be restyled without
a two-hour re-run.

Body figures are p2_fig1..p2_fig6; appendix figures are p2_appN_*.

    python -m experiments.figures            # all of them, skipping what is missing
    python -m experiments.figures fig5 fig6  # a subset, by name
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import ARTIFACTS, R_GRID, R_HEADLINE, RESULTS
from src.costs import elkan_p_star
from src.plotting import (
    AXIS,
    CAT,
    CRITICAL,
    GRID,
    INK,
    INK_2,
    MUTED,
    SEQ_CMAP,
    errorbar_ladder,
    save_fig,
)


def result(exp_id):
    path = RESULTS / f"{exp_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run python -m experiments.{exp_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_label(exp_id):
    p = result(exp_id)
    return p, {r["label"]: r for r in p["rows"]}


# ------------------------------------------------------------------ body figures


def fig1_ladder():
    """Complexity ladder: ranking gain against interpretability cost."""
    p, by = rows_by_label("exp01_ladder")
    rows = p["rows"]
    labels = [r["model"] for r in rows]
    means = [r["cv"]["roc_auc_mean"] for r in rows]
    sds = [r["cv"]["roc_auc_sd"] for r in rows]
    costs = [r["interp_cost"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    errorbar_ladder(ax1, labels, means, sds,
                    "a. Ranking quality by rung (25 fits, mean +/- SD)",
                    ref=by["rung1_pay0_rule"]["cv"]["roc_auc_mean"])
    ax1.set_xlim(0.45, 0.82)
    ax1.text(by["rung1_pay0_rule"]["cv"]["roc_auc_mean"] + 0.004, 0.15,
             "one rule on PAY_0", fontsize=8, color=MUTED, rotation=90, va="bottom")

    # Short, unambiguous names: two rungs are both "XGBoost" and two both read a
    # single PAY_0 column, so the model string alone does not identify a point.
    short = {"rung0_majority": "Majority", "rung1_pay0_rule": "PAY_0 rule",
             "rung1b_pay0_score": "PAY_0 score", "rung2_logistic": "Logistic",
             "rung3_tree": "Tree", "rung4_forest": "Forest",
             "rung5_xgb_default": "XGB default", "rung6_xgb_tuned": "XGB tuned"}
    # The two single-PAY_0 rungs share cost 1, and tree/XGB-default collide near 10^2.
    offsets = {"rung1b_pay0_score": (7, 4), "rung3_tree": (7, -11),
               "rung5_xgb_default": (7, 4)}
    x = [c if c else np.nan for c in costs]
    ax2.scatter(x, means, s=42, color=CAT[0], zorder=4)
    for r, xi, yi in zip(rows, x, means):
        if np.isfinite(xi):
            ax2.annotate(short.get(r["label"], r["label"]), (xi, yi),
                         textcoords="offset points",
                         xytext=offsets.get(r["label"], (7, -3)),
                         fontsize=8, color=INK_2)
    ax2.set_xscale("log")
    ax2.set_xlim(0.5, 6e6)
    ax2.set_xlabel("Interpretability cost (parameters or nodes to read, log scale)")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("b. The accuracy/explainability trade-off")
    ax2.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, "p2_fig1_complexity_ladder")


def fig2_roc_pr():
    """ROC and PR on the hold-out: tuned XGBoost against the ladder baselines."""
    y = np.load(ARTIFACTS / "y_test.npy")
    series = [("xgb_tuned", "XGBoost (tuned)", CAT[0], 2.0),
              ("rung5_xgb_default", "XGBoost (library defaults)", CAT[1], 1.4),
              ("rung1_pay0_baseline", "PAY_0 single rule", CAT[2], 1.4)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for label, name, colour, lw in series:
        path = ARTIFACTS / f"test_proba_{label}.npy"
        if not path.exists():
            continue
        proba = np.load(path)
        fpr, tpr, _ = roc_curve(y, proba)
        ax1.plot(fpr, tpr, color=colour, lw=lw,
                 label=f"{name} (AUC {roc_auc_score(y, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        if len(np.unique(proba)) <= 5:
            # A depth-1 tree emits two distinct scores, so it has two attainable operating
            # points. Joining them would interpolate through (recall, precision) pairs the
            # rule cannot reach -- invalid in PR space (Davis & Goadrich, 2006) -- and would
            # read as a far larger area than the average precision printed beside it.
            ax2.plot(rec, prec, "o", color=colour, markersize=6,
                     label=f"{name} (AP {ap:.3f}, attainable points)")
        else:
            ax2.plot(rec, prec, color=colour, lw=lw, label=f"{name} (AP {ap:.3f})")

    ax1.plot([0, 1], [0, 1], color=AXIS, lw=1.0, ls="--", zorder=1)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("a. ROC, held-out rows")
    ax1.legend(loc="lower right")
    ax2.axhline(float(y.mean()), color=AXIS, lw=1.0, ls="--", zorder=1)
    ax2.text(0.62, y.mean() + 0.012, f"prevalence {y.mean():.3f}", fontsize=8, color=MUTED)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("b. Precision-recall, held-out rows")
    ax2.legend(loc="upper right")
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
    return save_fig(fig, "p2_fig2_roc_pr")


def fig3_reliability():
    """Reliability: reweighting moves calibration even where it leaves ranking alone."""
    p = result("exp09_calibration")
    curves = p["config"]["reliability_curves"]
    show = [("A_spw1_raw", "spw = 1, raw", CAT[0]),
            ("C_spw_raw", "spw = 3.52, raw", CAT[1]),
            ("A_spw1_isotonic", "spw = 1, isotonic", CAT[2])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1, 0.62], "wspace": 0.3})
    by = {r["label"]: r for r in p["rows"]}
    ax1.plot([0, 1], [0, 1], color=AXIS, lw=1.0, ls="--", zorder=1)
    for key, name, colour in show:
        c = curves.get(key)
        if c is None:
            continue
        xs = [b["mean_pred"] for b in c if b["mean_pred"] is not None]
        ys = [b["frac_pos"] for b in c if b["mean_pred"] is not None]
        ax1.plot(xs, ys, "o-", color=colour, lw=1.6, markersize=4,
                 label=f"{name}  (ECE {by[key]['ece']:.3f})")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed default rate")
    ax1.set_title("a. Reliability, 10 bins, out-of-fold")
    ax1.legend(loc="upper left")

    keys = [k for k, _, _ in show if k in by]
    ax2.bar(np.arange(len(keys)), [by[k]["cv"]["brier_mean"] for k in keys],
            yerr=[by[k]["cv"]["brier_sd"] or 0 for k in keys],
            color=[c for k, _, c in show if k in by], width=0.6, zorder=3,
            error_kw={"ecolor": INK_2, "elinewidth": 1.1, "capsize": 4})
    for i, k in enumerate(keys):
        ax2.text(i, by[k]["cv"]["brier_mean"] + 0.004, f"{by[k]['cv']['brier_mean']:.4f}",
                 ha="center", fontsize=8, color=INK_2)
    ax2.set_xticks(np.arange(len(keys)), [n for k, n, _ in show if k in by],
                   rotation=20, ha="right")
    ax2.set_ylabel("Brier score (lower is better)")
    ax2.set_title("b. Brier score")
    ax2.grid(axis="x", visible=False)
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
    return save_fig(fig, "p2_fig3_reliability")


def fig4_cost():
    """Cost against threshold, with Elkan's analytic optimum overlaid."""
    p = result("exp10_cost")
    curves = p["config"]["cost_curves"]
    by = {r["label"]: r for r in p["rows"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"wspace": 0.26})
    shades = SEQ_CMAP(np.linspace(0.25, 0.95, len(R_GRID)))
    for colour, r in zip(shades, R_GRID):
        c = curves.get(f"r{r}_flat")
        if c is None:
            continue
        grid, cost = np.asarray(c["grid"]), np.asarray(c["cost"], dtype=float)
        ax1.plot(grid, cost / cost.max(), color=colour, lw=1.5, label=f"r = {r}")
        tau = by[f"r{r}_flat"]["tau_empirical"]
        ax1.plot([tau], [cost.min() / cost.max()], "o", color=colour, markersize=5,
                 zorder=5)
    ax1.set_xlim(0, 0.8)
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Total cost (scaled to each curve's maximum)")
    ax1.set_title("a. Out-of-fold cost curves (dots: optimum)")
    ax1.legend(loc="upper right", ncol=2)

    rs = [r for r in R_GRID if f"r{r}_flat" in by]
    ax2.plot(rs, [elkan_p_star(r) for r in rs], "-", color=INK_2, lw=1.6,
             label="Elkan (2001) p* = 1/(1+r)")
    ax2.plot(rs, [by[f"r{r}_flat"]["tau_empirical"] for r in rs], "o", color=CAT[0],
             markersize=6, label="empirical OOF optimum, raw")
    if f"r{R_HEADLINE}_flat_isotonic" in by:
        ax2.plot(rs, [by[f"r{r}_flat_isotonic"]["tau_empirical"] for r in rs], "s",
                 color=CAT[2], markersize=5, label="empirical OOF optimum, isotonic")
    ax2.set_xscale("log")
    ax2.set_xticks(rs, [str(r) for r in rs])
    ax2.set_xlabel("Cost ratio r = C_FN / C_FP")
    ax2.set_ylabel("Cost-optimal threshold")
    ax2.set_title("b. Optimum vs Elkan's analytic curve")
    ax2.legend(loc="upper right")
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
    return save_fig(fig, "p2_fig4_cost_threshold")


def _shap_frame():
    sv = np.load(ARTIFACTS / "shap_values_test.npy")
    X = pd.read_csv(ARTIFACTS / "shap_explain_X.csv")
    return sv, X


def fig5_beeswarm():
    """SHAP beeswarm on held-out decisions."""
    import shap

    sv, X = _shap_frame()
    base = np.load(ARTIFACTS / "shap_base_values_test.npy")
    expl = shap.Explanation(values=sv, base_values=base, data=X.to_numpy(),
                            feature_names=list(X.columns))
    fig = plt.figure(figsize=(7.6, 5.6))
    # Keeps SHAP's diverging default rather than the house sequential map: a beeswarm
    # encodes feature value on colour, and a single-hue ramp makes the low end vanish.
    shap.plots.beeswarm(expl, max_display=14, show=False, color_bar_label="Feature value")
    ax = plt.gca()
    ax.set_xlabel("SHAP value (log-odds contribution)")
    ax.set_title("Per-client attribution on 2,000 held-out decisions", loc="left",
                 fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save_fig(fig, "p2_fig5_shap_beeswarm")


def fig6_dependence():
    """PAY_0 dependence coloured by PAY_2 -- the model's internals against the EDA gradient."""
    sv, X = _shap_frame()
    p = result("exp11_shap")
    profile = p["config"]["pay0_profile"]
    j = list(X.columns).index("PAY_0")

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.18, 0.18, len(X))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    sc = ax1.scatter(X["PAY_0"] + jitter, sv[:, j], c=X["PAY_2"], cmap=SEQ_CMAP,
                     s=11, alpha=0.75, linewidths=0, zorder=3)
    ax1.axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    cb = fig.colorbar(sc, ax=ax1, pad=0.02)
    cb.set_label("PAY_2 (previous month's status)", fontsize=9, color=INK_2)
    cb.outline.set_visible(False)
    ax1.set_xlabel("PAY_0 (most recent repayment status, jittered)")
    ax1.set_ylabel("SHAP value for PAY_0 (log-odds)")
    ax1.set_title("a. The model's response to PAY_0")
    ax1.set_xticks(sorted(int(c) for c in profile))

    codes = sorted(int(c) for c in profile)
    rate = [100 * profile[str(c)]["observed_default_rate"] for c in codes]
    mshap = [profile[str(c)]["mean_shap_logodds"] for c in codes]
    # Codes 4, 5 and 8 hold a handful of clients each, so their rates are one or two
    # people wide. Faded, so the eye does not read them as evidence.
    sparse = [profile[str(c)]["n"] < 20 for c in codes]
    ax2.bar(np.arange(len(codes)), rate, width=0.6, zorder=3,
            color=[CAT[0]] * len(codes),
            alpha=None, label="observed default rate (%)")
    for bar, thin in zip(ax2.patches, sparse):
        if thin:
            bar.set_alpha(0.28)
    ax2.set_xticks(np.arange(len(codes)),
                   [f"{c}\nn={profile[str(c)]['n']:,}" for c in codes], fontsize=8)
    ax2.set_ylabel("Observed default rate (%)")
    ax2.set_xlabel("PAY_0 code")
    ax2b = ax2.twinx()
    ax2b.plot(np.arange(len(codes)), mshap, "o-", color=CRITICAL, lw=1.6, markersize=5,
              zorder=4, label="mean SHAP (log-odds)")
    ax2b.set_ylabel("Mean SHAP for PAY_0 (log-odds)", color=CRITICAL)
    ax2b.grid(False)
    # Headroom so the legend cannot sit on the code-2 peak.
    ax2.set_ylim(0, 128)
    lo, hi = min(mshap), max(mshap)
    ax2b.set_ylim(lo - 0.1 * (hi - lo), hi + 0.42 * (hi - lo))
    dense = [c for c in codes if profile[str(c)]["n"] >= 20]
    rho_dense = spearmanr([profile[str(c)]["mean_shap_logodds"] for c in dense],
                          [profile[str(c)]["observed_default_rate"] for c in dense]).statistic
    ax2.set_title(f"b. Convergent validation: Spearman rho {rho_dense:.3f} "
                  f"over the {len(dense)} codes with n >= 20")
    ax2.grid(axis="x", visible=False)
    ax2.set_axisbelow(True)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")
    return save_fig(fig, "p2_fig6_shap_pay0_dependence")


# ------------------------------------------------------------------ appendix figures


def app1_optuna():
    """Optimisation history and hyperparameter importances."""
    import optuna

    db = ARTIFACTS / "optuna_study.db"
    study = optuna.load_study(study_name="xgb_roc_auc", storage=f"sqlite:///{db}")
    done = [t for t in study.trials if t.value is not None]
    values = [t.value for t in done]
    running_best = np.maximum.accumulate(values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.scatter(range(len(values)), values, s=16, color=MUTED, zorder=3, label="trial")
    ax1.plot(running_best, color=CAT[0], lw=1.8, zorder=4, label="best so far")
    pruned = [i for i, t in enumerate(study.trials)
              if t.state == optuna.trial.TrialState.PRUNED]
    ax1.set_xlabel(f"Completed trial ({len(pruned)} further trials pruned)")
    ax1.set_ylabel("Mean 5-fold ROC-AUC")
    ax1.set_ylim(min(values) - 0.002, max(values) + 0.002)
    ax1.set_title("a. TPE optimisation history")
    ax1.legend(loc="lower right")

    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}
    if imp:
        names = list(imp)[::-1]
        ax2.barh(np.arange(len(names)), [imp[n] for n in names], color=CAT[0],
                 height=0.6, zorder=3)
        ax2.set_yticks(np.arange(len(names)), names)
        ax2.set_xlabel("fANOVA importance (share of objective variance)")
    ax2.set_title("b. Hyperparameter importance")
    ax2.grid(axis="y", visible=False)
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
    return save_fig(fig, "p2_app1_optuna")


def app2_learning_curve():
    """Train and validation ROC-AUC against training fraction."""
    p = result("exp08_learning_curve")
    rows = sorted(p["rows"], key=lambda r: r["fraction"])
    n = [r["n_train_rows_mean"] for r in rows]
    va = np.array([r["cv"]["roc_auc_mean"] for r in rows])
    va_sd = np.array([r["cv"]["roc_auc_sd"] for r in rows])
    tr = np.array([r["train_roc_auc_mean"] for r in rows])
    tr_sd = np.array([r["train_roc_auc_sd"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(n, tr, "o-", color=CAT[1], lw=1.6, markersize=5, label="training fold")
    ax.fill_between(n, tr - tr_sd, tr + tr_sd, color=CAT[1], alpha=0.16, lw=0)
    ax.plot(n, va, "o-", color=CAT[0], lw=1.8, markersize=5, label="validation fold")
    ax.fill_between(n, va - va_sd, va + va_sd, color=CAT[0], alpha=0.18, lw=0)
    ax.set_xlabel("Training rows used")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning curve, 5 fractions x 5 seeds (mean +/- SD)")
    ax.legend(loc="center right")
    ax.set_axisbelow(True)
    return save_fig(fig, "p2_app2_learning_curve")


def app3_recency():
    """How much repayment history is actually worth collecting."""
    p = result("exp07_recency")
    rows = sorted(p["rows"], key=lambda r: r["k_months"])
    full = rows[-1]["cv"]["roc_auc_mean"]
    sd = rows[-1]["cv"]["roc_auc_sd"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    k = [r["k_months"] for r in rows]
    m = [r["cv"]["roc_auc_mean"] for r in rows]
    ax.errorbar(k, m, yerr=[r["cv"]["roc_auc_sd"] for r in rows], fmt="o-",
                color=CAT[0], ecolor=INK_2, elinewidth=1.1, capsize=4, lw=1.8,
                markersize=5, zorder=4)
    ax.axhspan(full - sd, full + sd, color=CAT[0], alpha=0.12, lw=0, zorder=1)
    ax.axhline(full, color=AXIS, lw=1.1, zorder=2)
    ax.text(1.05, full + 0.0004, "all six months +/- 1 SD", fontsize=8, color=MUTED)
    ax.set_xticks(k)
    ax.set_xlabel("Most recent months of history retained")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Recency ablation (25 repeated-CV fits)")
    ax.set_axisbelow(True)
    return save_fig(fig, "p2_app3_recency")


def app4_groups():
    """Leave-one-group-out against only-one-group."""
    p = result("exp06_groups")
    by = {r["label"]: r for r in p["rows"]}
    allg = by["all_groups"]["cv"]["roc_auc_mean"]
    drop = sorted([r for r in p["rows"] if r["label"].startswith("drop_")],
                  key=lambda r: r["cv"]["roc_auc_mean"])
    only = sorted([r for r in p["rows"] if r["label"].startswith("only_")],
                  key=lambda r: r["cv"]["roc_auc_mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    errorbar_ladder(ax1, [r["label"].replace("drop_", "") for r in drop],
                    [r["cv"]["roc_auc_mean"] for r in drop],
                    [r["cv"]["roc_auc_sd"] for r in drop],
                    "a. Leave one group out", ref=allg)
    errorbar_ladder(ax2, [r["label"].replace("only_", "") for r in only],
                    [r["cv"]["roc_auc_mean"] for r in only],
                    [r["cv"]["roc_auc_sd"] for r in only],
                    "b. One group only", ref=allg)
    ax1.text(allg + 0.0015, 0.1, "all groups", fontsize=8, color=MUTED, rotation=90)
    return save_fig(fig, "p2_app4_feature_groups")


def app5_waterfalls():
    """The most confident true positive beside the worst false negative."""
    import shap

    sv, X = _shap_frame()
    base = np.load(ARTIFACTS / "shap_base_values_test.npy")
    cases = result("exp11_shap")["config"]["waterfall_cases"]

    fig = plt.figure(figsize=(12, 5.2))
    for i, (key, title) in enumerate((("true_positive", "a. Most confident true positive"),
                                      ("false_negative", "b. Worst false negative"))):
        idx = cases[key]["row"]
        plt.subplot(1, 2, i + 1)
        expl = shap.Explanation(
            values=sv[idx], base_values=float(np.atleast_1d(base)[0]),
            data=X.iloc[idx].to_numpy(), feature_names=list(X.columns))
        shap.plots.waterfall(expl, max_display=11, show=False)
        ax = plt.gca()
        ax.set_title(f"{title} (p = {cases[key]['proba']:.3f}, y = {cases[key]['y']})",
                     loc="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    return save_fig(fig, "p2_app5_shap_waterfalls")


def app6_fairness():
    """Per-subgroup rates with Wilson intervals, and error rates at the frozen threshold."""
    p = result("exp13_fairness")
    detail = p["config"]["subgroup_detail"]
    attrs = ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"]

    # Portrait, one attribute per row. A 2x4 landscape layout has to be scaled to about
    # half size to fit a page width, which puts the tick labels below 4pt in print.
    fig, axes = plt.subplots(4, 2, figsize=(9, 12.5))
    for col, attr in enumerate(attrs):
        groups = detail[f"full|{attr}"]
        names = [g["group"] for g in groups]
        x = np.arange(len(groups))
        rate = np.array([g["observed_default_rate_pct"] for g in groups])
        lo = np.array([g["wilson_lo_pct"] for g in groups])
        hi = np.array([g["wilson_hi_pct"] for g in groups])
        sel = np.array([100 * g["selection_rate"] for g in groups])

        ax = axes[col, 0]
        ax.bar(x - 0.19, rate, width=0.36, color=CAT[0], zorder=3, label="observed rate")
        ax.errorbar(x - 0.19, rate, yerr=[rate - lo, hi - rate], fmt="none",
                    ecolor=INK_2, elinewidth=1.0, capsize=3, zorder=4)
        ax.bar(x + 0.19, sel, width=0.36, color=CAT[1], zorder=3, label="model flag rate")
        ax.set_xticks(x, [f"{n}\nn={g['n']:,}" for n, g in zip(names, groups)],
                      rotation=25, ha="right", fontsize=7.5)
        ax.set_title(f"{attr} -- observed vs flagged", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_ylabel("Percent")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

        ax = axes[col, 1]
        for name, key, colour in (("recall", "recall", CAT[0]), ("FPR", "fpr", CAT[3]),
                                  ("ROC-AUC", "roc_auc", CAT[2])):
            vals = [g[key] for g in groups]
            ax.plot(x, vals, "o-", color=colour, lw=1.5, markersize=4, label=name)
        ax.set_xticks(x, names, rotation=25, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(f"{attr} -- at the frozen r={R_HEADLINE} threshold", fontsize=10)
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Subgroup performance, full model, out-of-fold training predictions",
                 x=0.005, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_fig(fig, "p2_app6_fairness")


def app7_importance_agreement():
    """Do the importance measures even agree on an ordering?"""
    p = result("exp12_stability")
    matrix = p["config"]["importance_agreement_matrix"]
    names = list(matrix)
    M = np.array([[matrix[a][b] for b in names] for a in names], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    im = ax1.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    ax1.set_yticks(np.arange(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if M[i, j] > 0.6 else INK)
    ax1.set_title("a. Spearman rho between importance measures")
    ax1.grid(False)
    cb = fig.colorbar(im, ax=ax1, pad=0.02, shrink=0.85)
    cb.outline.set_visible(False)

    rows = sorted(p["rows"], key=lambda r: r["shap_rank"])[:14]
    y = np.arange(len(rows))[::-1]
    ax2.barh(y, [r["rank_sd_across_seeds"] for r in rows], height=0.6,
             color=[CRITICAL if r["label"].startswith("BILL_AMT") else CAT[0]
                    for r in rows], zorder=3)
    ax2.set_yticks(y, [r["label"] for r in rows], fontsize=8)
    ax2.set_xlabel("SD of SHAP rank across 5 reseeded refits (positions)")
    ax2.set_title("b. Where the explanation is unstable (BILL_AMT in red)")
    ax2.grid(axis="y", visible=False)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, "p2_app7_importance_agreement")


def app8_engineered_encoding():
    """The two ablations the body only gets a sentence for."""
    eng = result("exp05_engineered")["rows"]
    enc = result("exp04_encoding")["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    errorbar_ladder(ax1, [r["label"] for r in eng],
                    [r["cv"]["roc_auc_mean"] for r in eng],
                    [r["cv"]["roc_auc_sd"] for r in eng],
                    "a. Engineered features against the raw 23",
                    ref=next(r["cv"]["roc_auc_mean"] for r in eng
                             if r["label"] == "raw_23"))
    errorbar_ladder(ax2, [r["label"] for r in enc],
                    [r["cv"]["roc_auc_mean"] for r in enc],
                    [r["cv"]["roc_auc_sd"] for r in enc],
                    "b. Encoding of the ordinal PAY_n columns",
                    ref=next(r["cv"]["roc_auc_mean"] for r in enc
                             if r["label"] == "native_int"))
    fig.tight_layout()
    return save_fig(fig, "p2_app8_engineered_encoding")


def app9_seed_variance():
    """Seed spread and the PAY_0 ablation, the two robustness checks."""
    p = result("exp14_final")
    cfg = p["config"]
    aucs = cfg["seed_test_roc_auc"]
    ab = cfg["pay0_ablation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.plot(np.arange(len(aucs)), aucs, "o", color=CAT[0], markersize=6, zorder=4)
    mean = float(np.mean(aucs))
    sd = cfg["seed_roc_auc_sd"]
    ax1.axhspan(mean - sd, mean + sd, color=CAT[0], alpha=0.14, lw=0, zorder=1)
    ax1.axhline(mean, color=AXIS, lw=1.1, zorder=2)
    ax1.set_xticks(np.arange(len(aucs)))
    ax1.set_xlabel("Refit (seed 42-51)")
    ax1.set_ylabel("Test ROC-AUC")
    ax1.set_title(f"a. Seed variance: {mean:.5f} +/- {sd:.5f}")

    labels = ["all 23 features", "PAY_0 dropped"]
    means = [ab["cv_roc_auc_full"], ab["cv_roc_auc_no_pay0"]]
    sds = [ab["cv_sd_full"], ab["cv_sd_full"]]
    ax2.bar(np.arange(2), means, yerr=sds, width=0.5, color=[CAT[0], CRITICAL], zorder=3,
            error_kw={"ecolor": INK_2, "elinewidth": 1.1, "capsize": 4})
    ax2.set_ylim(min(means) - 0.03, max(means) + 0.01)
    ax2.set_xticks(np.arange(2), labels)
    ax2.set_ylabel("CV ROC-AUC (25 fits)")
    ax2.set_title(f"b. PAY_0 ablation: {ab['paired_fold_test']['mean_delta']:+.4f} paired")
    ax2.grid(axis="x", visible=False)
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, "p2_app9_robustness")


BUILDERS = {
    "fig1": fig1_ladder, "fig2": fig2_roc_pr, "fig3": fig3_reliability,
    "fig4": fig4_cost, "fig5": fig5_beeswarm, "fig6": fig6_dependence,
    "app1": app1_optuna, "app2": app2_learning_curve, "app3": app3_recency,
    "app4": app4_groups, "app5": app5_waterfalls, "app6": app6_fairness,
    "app7": app7_importance_agreement, "app8": app8_engineered_encoding,
    "app9": app9_seed_variance,
}


def main(names=None):
    todo = names or list(BUILDERS)
    built, skipped = [], []
    for name in todo:
        try:
            BUILDERS[name]()
            built.append(name)
        except FileNotFoundError as e:
            skipped.append(f"{name}: {e}")
        finally:
            plt.close("all")
    print(f"\nbuilt {len(built)}: {', '.join(built)}")
    for s in skipped:
        print(f"skipped {s}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
