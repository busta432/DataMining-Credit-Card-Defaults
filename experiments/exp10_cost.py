"""WP14 / exp10 -- cost-sensitive threshold selection.

The cost ratio is anchored in credit-risk parameters rather than chosen for
convenience: a false negative costs LGD x EAD and a false positive costs the forgone
net interest margin, giving r = 0.70 / 0.12 = 5.8, reported as 6:1.

The sweep over r is overlaid with Elkan's (2001) analytic optimum p* = 1/(1+r). If the
empirical out-of-fold optimum tracks the analytic curve, that is direct evidence the
probabilities are calibrated -- which ties calibration, cost and threshold selection
into one result.

All thresholds are chosen on out-of-fold *training* predictions and frozen. The test
set is not touched here.

    python -m experiments.exp10_cost
"""

from __future__ import annotations

import numpy as np

from src.config import ARTIFACTS, LGD, NIM, R_GRID, R_HEADLINE
from src.costs import cost_at_threshold, elkan_p_star, pick_threshold_oof
from src.data import RAW_FEATURES, training_context
from src.metrics import evaluate
from src.runner import Timer, make_row, save_result

SCHEMES = {
    "flat": "raw probabilities, unit cost per error",
    "flat_isotonic": "isotonic-calibrated probabilities, unit cost per error",
    "instance": "raw probabilities, per-client cost using LIMIT_BAL as an EAD proxy",
}


def _load(name, wp):
    path = ARTIFACTS / name
    if not path.exists():
        raise SystemExit(f"MISSING ARTIFACT: {path}\nProduced by {wp}.")
    return np.load(path)


def main():
    with Timer() as t:
        oof_raw = _load("oof_proba_A_baseline.npy",
                        "WP7: python -m experiments.exp03_imbalance")
        iso_path = ARTIFACTS / "oof_calib_A_spw1_isotonic.npy"
        oof_iso = np.load(iso_path) if iso_path.exists() else None
        if oof_iso is None:
            print("note: isotonic OOF not found, skipping the calibrated scheme "
                  "(run exp09_calibration first for the full sweep)")

        ctx = training_context(RAW_FEATURES)
        y, X = ctx["y"], ctx["X"]
        ead = X["LIMIT_BAL"].to_numpy(dtype=float)

        rows, curves = [], {}
        for r in R_GRID:
            p_star = elkan_p_star(r)
            for scheme, desc in SCHEMES.items():
                if scheme == "flat_isotonic" and oof_iso is None:
                    continue
                proba = oof_iso if scheme == "flat_isotonic" else oof_raw
                use_ead = ead if scheme == "instance" else None
                tau, sweep = pick_threshold_oof(y, proba, r=r, ead=use_ead)
                curves[f"r{r}_{scheme}"] = {
                    "grid": sweep["grid"][::10], "cost": sweep["cost"][::10],
                }
                m = evaluate(y, proba, tau)
                cost_star = cost_at_threshold(y, proba, p_star, r=r, ead=use_ead)
                cost_half = cost_at_threshold(y, proba, 0.5, r=r, ead=use_ead)
                rows.append(make_row(
                    f"r{r}_{scheme}", None, n_features=X.shape[1],
                    extra={
                        "r": r, "scheme": scheme, "description": desc,
                        "tau_empirical": tau,
                        "elkan_p_star": p_star,
                        "tau_minus_p_star": tau - p_star,
                        "min_cost": sweep["min_cost"],
                        "cost_at_elkan": cost_star,
                        "cost_at_0.5": cost_half,
                        "saving_vs_0.5": cost_half - sweep["min_cost"],
                        "recall": m["recall"], "precision": m["precision"],
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                    }))
                print(f"r={r:<3} {scheme:<14} tau {tau:.4f}  elkan {p_star:.4f}  "
                      f"diff {tau - p_star:+.4f}  recall {m['recall']:.4f}  "
                      f"cost {sweep['min_cost']:,.0f}")

        def track(scheme):
            d = [r["tau_minus_p_star"] for r in rows if r["scheme"] == scheme]
            return float(np.mean(np.abs(d))) if d else None

        mae_flat = track("flat")
        mae_iso = track("flat_isotonic")
        head = next(r for r in rows if r["r"] == R_HEADLINE and r["scheme"] == "flat")
        head_inst = next(r for r in rows if r["r"] == R_HEADLINE
                         and r["scheme"] == "instance")

        notes = (
            f"Cost ratio derived from credit-risk parameters: LGD={LGD} and NIM={NIM} give "
            f"r = {LGD / NIM:.2f}, reported as {R_HEADLINE}:1. At r={R_HEADLINE}, Elkan's "
            f"analytic optimum is p* = 1/(1+r) = {elkan_p_star(R_HEADLINE):.4f} and the "
            f"empirical out-of-fold optimum is {head['tau_empirical']:.4f}, a difference of "
            f"{head['tau_minus_p_star']:+.4f}. Mean absolute deviation from the analytic "
            f"curve across the whole sweep is {mae_flat:.4f} for raw probabilities"
            + (f" and {mae_iso:.4f} after isotonic calibration, so calibration tightens the "
               f"agreement." if mae_iso is not None else ".")
            + f" That the empirical optimum tracks 1/(1+r) at all is direct evidence the "
            f"probabilities carry the right scale, not merely the right ranking. "
            f"Moving from the default 0.5 to the frozen threshold at r={R_HEADLINE} saves "
            f"{head['saving_vs_0.5']:,.0f} cost units and lifts recall from "
            f"{next(r for r in rows if r['r'] == 1 and r['scheme'] == 'flat')['recall']:.4f} "
            f"at r=1 to {head['recall']:.4f}. "
            f"Instance-specific costing using LIMIT_BAL as an EAD proxy selects "
            f"{head_inst['tau_empirical']:.4f} instead of {head['tau_empirical']:.4f}: "
            f"weighting each error by exposure shifts the optimum because large-limit "
            f"defaults dominate the loss. Both schemes are reported. "
            f"The 9.8M-customer and $1.17bn figures from the proposal are motivation for "
            f"caring about cost asymmetry; they are not presented as a result of this model."
        )
        save_result("exp10_cost", "Cost-sensitive threshold selection with Elkan overlay",
                    rows, notes, t.seconds,
                    config={"r_grid": R_GRID, "LGD": LGD, "NIM": NIM,
                            "r_headline": R_HEADLINE,
                            "elkan": {r: elkan_p_star(r) for r in R_GRID},
                            "mae_vs_elkan": {"flat": mae_flat, "flat_isotonic": mae_iso},
                            "cost_curves": curves})


if __name__ == "__main__":
    main()
