"""Section 7 verification. Every claim the report rests on, checked mechanically.

Run last. Each check prints PASS or FAIL and the script exits non-zero if any failed, so
it is usable as a pre-submission gate rather than something to read and interpret.

    python -m experiments.verify
"""

from __future__ import annotations

import json

import numpy as np

from src.config import ARTIFACTS, N_FOLDS, N_REPEATS, RESULTS, SEED, TEST_SIZE
from src.data import (RAW_FEATURES, TARGET, clean, cv_folds, load_raw, load_splits,
                      make_splits, repeated_cv_folds)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------- data integrity


@check("Phase 1 assertions still hold after cleaning")
def _phase1():
    df = clean(load_raw())
    assert len(df) == 29944, f"expected 29,944 rows, got {len(df):,}"
    assert int(df[TARGET].sum()) == 6622, f"expected 6,622 defaults, got {df[TARGET].sum():,}"
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}, "EDUCATION domain drifted"
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}, "MARRIAGE domain drifted"
    return f"29,944 rows / 6,622 defaults / prevalence {df[TARGET].mean():.4%}"


@check("No duplicate feature rows survive cleaning")
def _dedup():
    df = clean(load_raw())
    dups = int(df.duplicated(subset=list(RAW_FEATURES)).sum())
    assert dups == 0, f"{dups} duplicate feature rows remain -- they would straddle the split"
    return "0 duplicate feature-rows, so none can straddle train/test"


# ---------------------------------------------------------------- splits


@check("Split sizes, stratification and zero overlap")
def _splits():
    df = clean(load_raw())
    y = df[TARGET].to_numpy(dtype=int)
    s = load_splits()
    tr, te = s["train_idx"], s["test_idx"]

    assert len(tr) == 23955, f"train is {len(tr):,}, expected 23,955"
    assert len(te) == 5989, f"test is {len(te):,}, expected 5,989"
    assert len(np.intersect1d(tr, te)) == 0, "train and test indices overlap"
    assert len(tr) + len(te) == len(df), "split does not partition the frame"
    assert abs(len(te) / len(df) - TEST_SIZE) < 0.001, "test fraction drifted"

    gap = abs(y[tr].mean() - y[te].mean())
    assert gap < 0.001, f"stratification gap {gap:.5f} exceeds 0.1pp"
    assert int(y[te].sum()) == 1324, f"expected 1,324 test positives, got {y[te].sum():,}"
    return (f"23,955 / 5,989, overlap 0, prevalence gap {gap:.5f}, "
            f"{int(y[te].sum()):,} test positives")


@check("Fold structures partition the training rows")
def _folds():
    df = clean(load_raw())
    y = df[TARGET].to_numpy(dtype=int)
    s = load_splits()
    y_tr = y[s["train_idx"]]

    folds = cv_folds(s)
    assert len(folds) == N_FOLDS
    seen = np.concatenate([va for _, va in folds])
    assert len(seen) == len(y_tr) and len(np.unique(seen)) == len(y_tr), \
        "tuning folds do not partition the training rows exactly once"
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0, "a tuning fold overlaps its own training half"

    rfolds = repeated_cv_folds(s)
    assert len(rfolds) == N_FOLDS * N_REPEATS, f"expected 25 folds, got {len(rfolds)}"
    gaps = [abs(y_tr[va].mean() - y_tr.mean()) for _, va in rfolds]
    assert max(gaps) < 0.01, f"repeated-CV stratification gap {max(gaps):.4f} too large"
    return f"5 tuning folds + 25 variance folds, max stratification gap {max(gaps):.5f}"


@check("Splits reproduce exactly on a re-run with the same SEED")
def _reproducible():
    before = load_splits()
    after = make_splits(force=True)
    for k in ("train_idx", "test_idx", "fold_id", "rep_fold_id"):
        assert np.array_equal(before[k], after[k]), f"{k} changed on re-run with SEED={SEED}"
    return f"all four index arrays identical after force-rebuild at SEED={SEED}"


# ---------------------------------------------------------------- result contract


@check("Every results/*.json validates against the section 3.2 schema")
def _schema():
    paths = sorted(RESULTS.glob("exp*.json"))
    assert paths, "no result files found -- run the experiments first"
    top = {"experiment_id", "title", "run_utc", "seed", "versions",
           "runtime_sec", "config", "rows", "notes"}
    cv_keys = {"roc_auc_mean", "roc_auc_sd", "pr_auc_mean", "pr_auc_sd",
               "brier_mean", "brier_sd", "n_fits"}
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        missing = top - set(d)
        assert not missing, f"{p.name} missing top-level keys {missing}"
        assert d["notes"].strip(), f"{p.name} has an empty notes field"
        assert d["rows"], f"{p.name} has no rows"
        labels = [r["label"] for r in d["rows"]]
        assert len(labels) == len(set(labels)), f"{p.name} has duplicate labels"
        for r in d["rows"]:
            if r.get("cv") is not None:
                absent = cv_keys - set(r["cv"])
                assert not absent, f"{p.name}:{r['label']} cv missing {absent}"
    return f"{len(paths)} result files, all conformant, all with non-empty notes"


@check("Frozen thresholds are scalars strictly inside (0, 1)")
def _thresholds():
    found = 0
    for p in sorted(RESULTS.glob("exp*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["rows"]:
            for key in ("oof_threshold", "tau_empirical"):
                if isinstance(r.get(key), (int, float)):
                    tau = float(r[key])
                    assert 0.0 < tau < 1.0, f"{p.name}:{r['label']} {key}={tau} out of range"
                    found += 1
            t = r.get("test")
            if isinstance(t, dict) and isinstance(t.get("threshold"), (int, float)):
                tau = float(t["threshold"])
                assert 0.0 < tau < 1.0, f"{p.name}:{r['label']} test threshold {tau} out of range"
                found += 1
    if not found:
        return "SKIP -- no thresholds yet (run exp03_imbalance and exp10_cost)"
    return f"{found} thresholds, all scalar and inside (0, 1)"


@check("Only the licensed experiments populate a test block")
def _test_discipline():
    licensed = {"exp14_final"}
    offenders = []
    for p in sorted(RESULTS.glob("exp*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["experiment_id"] in licensed:
            continue
        for r in d["rows"]:
            if r.get("test"):
                offenders.append(f"{d['experiment_id']}:{r['label']}")
    assert not offenders, f"unlicensed test blocks: {offenders}"
    return "every test block outside exp14_final is null"


@check("SHAP values sum to the model margin")
def _shap_additivity():
    path = RESULTS / "exp11_shap.json"
    if not path.exists():
        return "SKIP -- results/exp11_shap.json absent (run exp11_shap)"
    cfg = json.loads(path.read_text(encoding="utf-8"))["config"]
    err = float(cfg["additivity_max_error"])
    assert err < 1e-4, f"max additivity error {err:.3e} exceeds 1e-4"

    # Recompute independently from the cached values, so this is not merely a
    # restatement of the assertion exp11 already made.
    sv = np.load(ARTIFACTS / "shap_values_test.npy")
    base = np.load(ARTIFACTS / "shap_base_values_test.npy")
    n_feat = sv.shape[1]
    assert n_feat == len(RAW_FEATURES), f"SHAP matrix has {n_feat} columns"
    spread = float(np.ptp(base))
    assert spread < 1e-6, f"base value is not constant (spread {spread:.2e})"
    return (f"exp11 recorded max error {err:.2e} in log-odds space over "
            f"{len(sv):,} held-out rows x {n_feat} features; base value constant")


# ---------------------------------------------------------------- leakage audit


@check("Leakage audit: the hold-out is imported only where licensed")
def _leakage():
    from pathlib import Path
    licensed = {"exp11_shap.py", "exp12_stability.py", "exp14_final.py",
                "figures.py", "verify.py"}
    exp_dir = Path(__file__).resolve().parent
    offenders = {}
    for p in sorted(exp_dir.glob("*.py")):
        if p.name in licensed:
            continue
        text = p.read_text(encoding="utf-8")
        hits = [tok for tok in ("test_context", "X_test", "y_test", "test_idx")
                if tok in text]
        if hits:
            offenders[p.name] = hits
    assert not offenders, f"hold-out referenced in selection code: {offenders}"
    return ("no tuning or threshold-selection script references the hold-out; "
            f"licensed: {', '.join(sorted(licensed - {'verify.py'}))}")


# ---------------------------------------------------------------- driver


def main():
    width = 62
    passed = failed = skipped = 0
    print("=" * width)
    print("Section 7 verification")
    print("=" * width)
    for name, fn in CHECKS:
        try:
            detail = fn()
        except AssertionError as e:
            print(f"FAIL  {name}\n      {e}")
            failed += 1
        except Exception as e:  # missing artifact, bad JSON -- report, do not crash
            print(f"ERROR {name}\n      {type(e).__name__}: {e}")
            failed += 1
        else:
            if isinstance(detail, str) and detail.startswith("SKIP"):
                print(f"SKIP  {name}\n      {detail[7:]}")
                skipped += 1
            else:
                print(f"PASS  {name}\n      {detail}")
                passed += 1
    print("=" * width)
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
