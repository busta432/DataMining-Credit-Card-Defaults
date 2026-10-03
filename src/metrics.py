"""Metric helpers. `wilson` is ported verbatim from Phase 1 Cell 20."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion; valid at small n and p near 0 or 1."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def brier(y_true, proba):
    return float(brier_score_loss(y_true, proba))


def evaluate(y_true, proba, threshold=0.5):
    """Ranking, calibration and thresholded-decision metrics for one prediction vector.

    `threshold` must already be frozen from out-of-fold data (see costs.pick_threshold_oof);
    this function never searches for one.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": brier(y_true, proba),
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def reliability(y_true, proba, n_bins=10):
    """Equal-width reliability curve, serialised so figures rebuild without refitting."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "mean_pred": float(proba[m].mean()) if m.any() else None,
            "frac_pos": float(y_true[m].mean()) if m.any() else None,
        })
    return out


# ---------------------------------------------------------------- DeLong


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(scores, n_pos):
    """Sun & Xu (2014) O(n log n) DeLong covariance. `scores` is (k, n) with positives first."""
    m = n_pos
    n = scores.shape[1] - m
    k = scores.shape[0]
    pos, neg = scores[:, :m], scores[:, m:]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(scores[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(y_true, proba_a, proba_b):
    """Paired DeLong test for two correlated ROC curves on the same sample.

    Valid only on a single held-out set -- never across CV folds, where the samples
    are not independent.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")  # positives first, stable
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    scores = np.vstack([np.asarray(proba_a, float)[order],
                        np.asarray(proba_b, float)[order]])
    aucs, cov = _fast_delong(scores, n_pos)
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    if var_diff <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
                "se": 0.0, "ci95": [diff, diff], "z": None, "p_value": None}
    se = float(np.sqrt(var_diff))
    z = diff / se
    p = float(2 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": diff,
        "se": se,
        "ci95": [diff - 1.96 * se, diff + 1.96 * se],
        "z": float(z),
        "p_value": p,
    }


def paired_fold_test(deltas):
    """Paired t-test on per-fold differences.

    Nadeau & Bengio (2003): repeated-CV t-tests are anti-conservative because the
    training folds overlap, so the p-value is a directional indicator, not a
    calibrated error rate.
    """
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return {"mean_delta": float(d.mean()), "sd": 0.0, "t": None, "p_value": None,
                "n": int(len(d))}
    t, p = stats.ttest_1samp(d, 0.0)
    return {"mean_delta": float(d.mean()), "sd": float(d.std(ddof=1)),
            "t": float(t), "p_value": float(p), "n": int(len(d))}
