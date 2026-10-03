"""Cost matrix, threshold selection and Elkan's analytic optimum.

The cost ratio is anchored in credit-risk parameters rather than chosen for convenience:
a false negative costs LGD x EAD (the unrecovered balance) and a false positive costs the
forgone net interest margin on a customer who would have repaid.
"""

from __future__ import annotations

import numpy as np

from .config import LGD, NIM, R_HEADLINE, THRESHOLD_GRID_N


def elkan_p_star(r):
    """Elkan (2001) cost-optimal decision threshold for a calibrated classifier.

    Acting is optimal when p * C_FN > (1 - p) * C_FP, i.e. p > 1 / (1 + r).
    """
    return 1.0 / (1.0 + r)


def cost_ratio_from_credit_params(lgd=LGD, nim=NIM):
    return lgd / nim


def cost_at_threshold(y_true, proba, tau, r=R_HEADLINE, ead=None):
    """Total misclassification cost at a fixed threshold.

    With `ead=None` every error costs the same (count x unit cost). With `ead` supplied
    -- LIMIT_BAL is the proxy used in WP14 -- cost is summed per client, so a missed
    default on a large limit costs more than one on a small limit.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba, dtype=float) >= tau).astype(int)
    fn = (y_true == 1) & (pred == 0)
    fp = (y_true == 0) & (pred == 1)
    if ead is None:
        return float(r * fn.sum() + fp.sum())
    ead = np.asarray(ead, dtype=float)
    return float(LGD * ead[fn].sum() + NIM * ead[fp].sum())


def threshold_grid(n=THRESHOLD_GRID_N):
    # Open interval: tau=0 predicts everyone default, tau=1 nobody. Neither is a decision.
    return np.linspace(1e-4, 1 - 1e-4, n)


def pick_threshold_oof(y_oof, proba_oof, r=R_HEADLINE, ead=None, grid=None):
    """Choose the cost-minimising threshold on out-of-fold training predictions.

    The returned scalar is frozen and applied unchanged to the test set. Sweeping the
    threshold against test labels is the single most common fatal flaw in this task.
    """
    grid = threshold_grid() if grid is None else np.asarray(grid, dtype=float)
    costs = np.array([cost_at_threshold(y_oof, proba_oof, t, r=r, ead=ead) for t in grid])
    tau = float(grid[int(np.argmin(costs))])
    return tau, {"grid": grid.tolist(), "cost": costs.tolist(),
                 "min_cost": float(costs.min()), "tau": tau}
