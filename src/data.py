"""Data loading, cleaning, feature engineering and splitting.

`load_raw` and `clean` are ported from Phase 1 Cells 4 and 24 with the assertions intact,
so Phase 2 provably operates on the same 29,944-row frame the proposal described.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from .config import ARTIFACTS, CV_SEED, N_FOLDS, N_REPEATS, SEED, TEST_SIZE

TARGET = "DEFAULT"

DEMOGRAPHIC = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_CHRONO = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]  # April -> September
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
BILL_CHRONO = BILL_COLS[::-1]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CONTINUOUS = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAYAMT_COLS
MONTH_LABEL = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]

RAW_FEATURES = ["LIMIT_BAL"] + DEMOGRAPHIC + PAY_COLS + BILL_COLS + PAYAMT_COLS

# WP10 ablates over this dict.
FEATURE_GROUPS = {
    "demographic": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
    "limit": ["LIMIT_BAL"],
    "pay_status": PAY_COLS,
    "bill": BILL_COLS,
    "pay_amt": PAYAMT_COLS,
}

SEX_LABEL = {1: "1 male", 2: "2 female"}
CLEAN_EDU = {1: "1 graduate school", 2: "2 university", 3: "3 high school", 4: "4 others"}
CLEAN_MAR = {1: "1 married", 2: "2 single", 3: "3 others"}
AGE_BANDS = [(0, 29, "<30"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 200, "50+")]

_RAW_CACHE = ARTIFACTS / "raw.csv"


def load_raw(use_cache=True):
    """Fetch UCI id=350, apply the programmatic rename, attach the target."""
    if use_cache and _RAW_CACHE.exists():
        raw = pd.read_csv(_RAW_CACHE)
    else:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=350)
        name_map = {
            r["name"]: r["description"]
            for _, r in repo.variables.iterrows()
            if isinstance(r.get("description"), str) and r["description"].strip()
        }
        raw = repo.data.features.rename(columns=name_map).copy()
        raw[TARGET] = repo.data.targets.iloc[:, 0].values
        raw.to_csv(_RAW_CACHE, index=False)

    assert list(raw.columns[:5]) == ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], \
        "Programmatic rename failed - check variables metadata"
    assert raw.shape == (30000, 24), f"Unexpected raw shape {raw.shape}"
    return raw


def clean(raw):
    """Phase 1 Cell 24: de-duplicate on features, fold undocumented codes into 'others'.

    De-duplication happens before any split, so identical feature-rows cannot straddle
    train and test and inflate every score.
    """
    feat_cols = [c for c in raw.columns if c != TARGET]
    df = raw.copy()
    df = df.drop_duplicates(subset=feat_cols, keep="first").reset_index(drop=True)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    assert len(df) == 29944, f"Expected 29,944 rows after de-duplication, got {len(df):,}"
    assert int(df[TARGET].sum()) == 6622, "Unexpected default count after cleaning"
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}, "EDUCATION outside documented domain"
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}, "MARRIAGE outside documented domain"
    assert int(df.isna().sum().sum()) == 0, "Nulls introduced by cleaning"
    assert not df[feat_cols].duplicated().any(), "Duplicates survived cleaning"
    return df


# ---------------------------------------------------------------- WP3: engineering

# Evenly spaced chronological month index; the OLS slope denominator is therefore
# a constant and the whole trend reduces to one dot product.
_X_MONTH = np.arange(6, dtype=float)
_X_CENTRED = _X_MONTH - _X_MONTH.mean()
_X_SS = float((_X_CENTRED**2).sum())  # 17.5


def _ols_slope(values):
    """Least-squares slope of each row against the chronological month index."""
    v = np.asarray(values, dtype=float)
    return (v - v.mean(axis=1, keepdims=True)) @ _X_CENTRED / _X_SS


ENGINEERED = (
    [f"UTIL_{i}" for i in range(1, 7)]
    + [f"PAY_RATIO_{i}" for i in range(1, 7)]
    + ["N_DELINQ", "MAX_DELINQ", "DELINQ_TREND", "BILL_TREND", "MONTHS_DORMANT"]
)


def engineer(df):
    """Add the 17 derived features WP9 benchmarks against the raw 23.

    Phase 1 predicted aggregation may *lose* signal by collapsing the recency gradient;
    WP9 tests that prediction rather than assuming these help.

    PAY_RATIO is undefined where the statement balance is zero or negative (1,930 rows
    carry a negative BILL_AMT, per Phase 1 section 1.6). Those entries become NaN rather
    than a sentinel: XGBoost learns a default direction for missing values natively, so
    NaN carries strictly more information than an arbitrary fill.
    """
    out = df.copy()
    limit = out["LIMIT_BAL"].to_numpy(dtype=float)
    assert (limit > 0).all(), "LIMIT_BAL must be positive for UTIL_n to be defined"

    bills = out[BILL_COLS].to_numpy(dtype=float)
    pays = out[PAYAMT_COLS].to_numpy(dtype=float)
    status = out[PAY_COLS].to_numpy(dtype=float)

    for i, col in enumerate(BILL_COLS, start=1):
        out[f"UTIL_{i}"] = bills[:, i - 1] / limit

    # Index-aligned with the bill of the same month, matching the column naming.
    safe_bills = np.where(bills > 0, bills, np.nan)
    ratios = pays / safe_bills
    for i in range(1, 7):
        out[f"PAY_RATIO_{i}"] = ratios[:, i - 1]

    out["N_DELINQ"] = (status >= 1).sum(axis=1)
    out["MAX_DELINQ"] = status.max(axis=1)
    out["MONTHS_DORMANT"] = (status == -2).sum(axis=1)
    out["DELINQ_TREND"] = _ols_slope(out[PAY_CHRONO].to_numpy(dtype=float))
    out["BILL_TREND"] = _ols_slope(out[BILL_CHRONO].to_numpy(dtype=float))

    added = [c for c in out.columns if c not in df.columns]
    assert set(added) == set(ENGINEERED), f"Engineered column set drifted: {added}"
    assert len(added) == 17, f"Expected 17 engineered columns, got {len(added)}"
    assert not np.isinf(out[added].to_numpy(dtype=float)).any(), "engineer() emitted inf"
    return out


def age_band(series):
    labels = pd.Series(index=series.index, dtype="object")
    for lo, hi, name in AGE_BANDS:
        labels[(series >= lo) & (series <= hi)] = name
    return labels


def xy(df, features=None):
    """Feature matrix / target split. Defaults to the raw 23 predictors."""
    cols = list(features) if features is not None else list(RAW_FEATURES)
    return df[cols].copy(), df[TARGET].to_numpy(dtype=int)


# ---------------------------------------------------------------- WP4: splits

SPLITS_PATH = ARTIFACTS / "splits.npz"


def make_splits(df=None, force=False):
    """Build and cache the 80/20 hold-out plus both fold structures.

    Written once by WP4 and read by every experiment, so all fourteen share identical
    indices and no experiment can silently reshuffle.
    """
    if SPLITS_PATH.exists() and not force:
        return load_splits()

    if df is None:
        df = clean(load_raw())
    y = df[TARGET].to_numpy(dtype=int)
    idx = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    y_train = y[train_idx]

    # Tuning / model-selection folds.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_id = np.empty(len(train_idx), dtype=np.int8)
    for f, (_, va) in enumerate(skf.split(train_idx, y_train)):
        fold_id[va] = f

    # Variance-reporting folds: distinct seed so repeats are not correlated with
    # the folds the tuner optimised against.
    rskf = RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=CV_SEED
    )
    rep_fold_id = np.empty((N_REPEATS, len(train_idx)), dtype=np.int8)
    for k, (_, va) in enumerate(rskf.split(train_idx, y_train)):
        rep_fold_id[k // N_FOLDS, va] = k % N_FOLDS

    np.savez_compressed(
        SPLITS_PATH,
        train_idx=train_idx,
        test_idx=test_idx,
        fold_id=fold_id,
        rep_fold_id=rep_fold_id,
    )
    print(f"saved -> {SPLITS_PATH}")
    return load_splits()


def load_splits():
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} is missing. Produce it with WP4: python -m src.data"
        )
    z = np.load(SPLITS_PATH)
    return {k: z[k] for k in z.files}


def cv_folds(splits):
    """Tuning folds as (train_positions, val_positions) pairs into the training frame."""
    fold_id = splits["fold_id"]
    return [(np.where(fold_id != f)[0], np.where(fold_id == f)[0])
            for f in range(N_FOLDS)]


def repeated_cv_folds(splits):
    """The 25 variance-reporting folds, as positions into the training frame."""
    rep = splits["rep_fold_id"]
    out = []
    for r in range(rep.shape[0]):
        for f in range(N_FOLDS):
            out.append((np.where(rep[r] != f)[0], np.where(rep[r] == f)[0]))
    return out


def frame(engineered=False):
    df = clean(load_raw())
    return engineer(df) if engineered else df


def training_context(features=None, engineered=False, df=None):
    """Training portion only, with both fold structures attached.

    Experiments WP5-WP13 use this exclusively. It cannot return test rows, which is what
    makes the leakage audit in section 7.5 a one-line grep: `X_test` should appear only
    in the final-evaluation and SHAP modules.
    """
    df = frame(engineered) if df is None else df
    s = load_splits()
    X, y = xy(df, features)
    tr = s["train_idx"]
    return {
        "df": df,
        "splits": s,
        "X": X.iloc[tr].reset_index(drop=True),
        "y": y[tr],
        "folds": cv_folds(s),
        "repeated_folds": repeated_cv_folds(s),
        "features": list(X.columns),
    }


def test_context(features=None, engineered=False, df=None):
    """Held-out rows. Licensed for WP14/WP15 (SHAP explains decisions, selects nothing)
    and WP18 (the single final-evaluation gate). Nothing else may import this."""
    df = frame(engineered) if df is None else df
    s = load_splits()
    X, y = xy(df, features)
    te = s["test_idx"]
    return {
        "df": df,
        "splits": s,
        "X_test": X.iloc[te].reset_index(drop=True),
        "y_test": y[te],
        "features": list(X.columns),
    }


if __name__ == "__main__":
    frame = clean(load_raw())
    frame = engineer(frame)
    s = make_splits(frame, force=True)
    ytr = frame[TARGET].to_numpy()[s["train_idx"]]
    yte = frame[TARGET].to_numpy()[s["test_idx"]]
    print(f"train {len(s['train_idx']):,} ({ytr.mean():.4%} default)  "
          f"test {len(s['test_idx']):,} ({yte.mean():.4%} default)")
    assert not set(s["train_idx"]) & set(s["test_idx"]), "train/test index overlap"
    print("splits ok")
