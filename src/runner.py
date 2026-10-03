"""The one evaluation path every experiment calls, plus the result-file writer.

Divergent CV loops are the main source of numbers that cannot be reconciled later, so
no experiment implements its own.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import ES_SLICE, RESULTS, SEED


def versions():
    import sklearn

    v = {"python": platform.python_version(), "numpy": np.__version__,
         "sklearn": sklearn.__version__}
    for mod in ("xgboost", "shap", "optuna", "imblearn", "pandas"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def _uses_early_stopping(est):
    return isinstance(est, XGBClassifier) and getattr(est, "early_stopping_rounds", None)


def fit_one(est, X_tr, y_tr, seed=SEED):
    """Fit a single estimator, carving the early-stopping slice out of the training data.

    The eval_set comes from an inner slice of this fold's *training* rows, so the outer
    validation fold is never seen by early stopping and the test set never at all.
    """
    if _uses_early_stopping(est):
        i_fit, i_es = train_test_split(
            np.arange(len(y_tr)), test_size=ES_SLICE, stratify=y_tr, random_state=seed
        )
        est.fit(X_tr.iloc[i_fit], y_tr[i_fit],
                eval_set=[(X_tr.iloc[i_es], y_tr[i_es])], verbose=False)
    else:
        est.fit(X_tr, y_tr)
    return est


def _fold_metrics(y_true, proba):
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def cv_evaluate(X, y, factory, folds, seed=SEED, collect_oof=True, return_models=False):
    """Fit `factory()` on every fold and report mean +/- SD across fits.

    `X` is a DataFrame and `y` a 1-D array, both already restricted to the training
    portion. `folds` are (train_positions, val_positions) pairs from data.cv_folds or
    data.repeated_cv_folds.

    The SD returned is across fits, not the standard error -- it is the quantity the
    report uses to decide whether two arms are distinguishable.
    """
    y = np.asarray(y).astype(int)
    per_fold, models = [], []
    oof_sum = np.zeros(len(y)) if collect_oof else None
    oof_n = np.zeros(len(y)) if collect_oof else None
    best_iters = []

    for k, (tr, va) in enumerate(folds):
        est = fit_one(factory(), X.iloc[tr], y[tr], seed=seed)
        proba = est.predict_proba(X.iloc[va])[:, 1]
        per_fold.append(_fold_metrics(y[va], proba))
        if collect_oof:
            oof_sum[va] += proba
            oof_n[va] += 1
        if isinstance(est, XGBClassifier) and getattr(est, "best_iteration", None) is not None:
            best_iters.append(int(est.best_iteration))
        if return_models:
            models.append(est)

    summary = {}
    for m in ("roc_auc", "pr_auc", "brier"):
        vals = np.array([f[m] for f in per_fold], dtype=float)
        summary[f"{m}_mean"] = float(vals.mean())
        summary[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary["n_fits"] = len(folds)

    out = {"cv": summary, "per_fold": per_fold}
    if best_iters:
        out["best_iteration_mean"] = float(np.mean(best_iters))
    if collect_oof:
        with np.errstate(invalid="ignore"):
            out["oof"] = np.where(oof_n > 0, oof_sum / np.maximum(oof_n, 1), np.nan)
    if return_models:
        out["models"] = models
    return out


def per_fold_metric(cv_out, metric="roc_auc"):
    """Per-fold values, for paired arm-vs-arm comparisons."""
    return np.array([f[metric] for f in cv_out["per_fold"]], dtype=float)


def make_row(label, cv_summary, params=None, n_features=None, test=None, extra=None):
    """Build one `rows[]` entry of the result contract.

    Any metric that does not apply is explicitly null, never omitted and never zero.
    """
    row = {
        "label": label,
        "params": params or {},
        "n_features": n_features,
        "cv": {
            "roc_auc_mean": cv_summary.get("roc_auc_mean"),
            "roc_auc_sd": cv_summary.get("roc_auc_sd"),
            "pr_auc_mean": cv_summary.get("pr_auc_mean"),
            "pr_auc_sd": cv_summary.get("pr_auc_sd"),
            "brier_mean": cv_summary.get("brier_mean"),
            "brier_sd": cv_summary.get("brier_sd"),
            "n_fits": cv_summary.get("n_fits"),
        } if cv_summary is not None else None,
        "test": test,
    }
    if extra:
        row.update(extra)
    return row


def save_result(experiment_id, title, rows, notes, runtime_sec, config=None, seed=SEED):
    """Write results/<experiment_id>.json against the section 3.2 contract."""
    if not notes or not str(notes).strip():
        raise ValueError("notes is mandatory; 'Nothing unusual' is acceptable, empty is not")
    payload = {
        "experiment_id": experiment_id,
        "title": title,
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "versions": versions(),
        "runtime_sec": round(float(runtime_sec), 1),
        "config": config or {},
        "rows": rows,
        "notes": notes,
    }
    path = RESULTS / f"{experiment_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")
    print(f"saved -> {path}")
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"{type(o)} is not JSON serialisable")


def load_result(experiment_id):
    path = RESULTS / f"{experiment_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run python -m experiments.{experiment_id}")
    return json.loads(path.read_text(encoding="utf-8"))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0

    @property
    def seconds(self):
        return time.perf_counter() - self.t0


def require_artifact(path, wp):
    """Fail loudly rather than silently recomputing -- that is how seeds drift."""
    if not path.exists():
        sys.exit(f"MISSING ARTIFACT: {path}\nProduced by {wp}. Run that first.")
    return path
