"""Model zoo. Every rung of the complexity ladder and every XGBoost variant is built here,
so no experiment constructs an estimator of its own and configurations cannot drift apart.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import ES_ROUNDS, ES_SLICE, N_ESTIMATORS_CAP, SEED, XGB_FIXED
from .data import PAY_COLS, RAW_FEATURES


def xgb_classifier(params=None, spw=1.0, early_stopping=True,
                   enable_categorical=False, seed=SEED):
    """XGBoost with the project's fixed infrastructure settings.

    `early_stopping_rounds` and `eval_metric` go in the constructor, not `.fit()` --
    the XGBoost 2.x/3.x API moved them. `enable_categorical` is pinned False because
    xgboost 3.4 defaults it True and shap then refuses interventional TreeSHAP.
    """
    p = dict(XGB_FIXED)
    p["enable_categorical"] = enable_categorical
    p["random_state"] = seed
    p["scale_pos_weight"] = spw
    if early_stopping:
        p["n_estimators"] = N_ESTIMATORS_CAP
        p["early_stopping_rounds"] = ES_ROUNDS
    if params:
        p.update(params)
    return XGBClassifier(**p)


def xgb_library_defaults(seed=SEED):
    """Rung 5: no tuning and no early stopping, so the tuning gain is isolated."""
    return XGBClassifier(**{**XGB_FIXED, "random_state": seed})


def majority():
    return DummyClassifier(strategy="most_frequent")


def single_rule():
    """Rung 1: one split on PAY_0. Recovers the Phase 1 benchmark that was lost with
    the uncommitted multivariate notebook."""
    return DecisionTreeClassifier(max_depth=1, random_state=SEED)


def logistic():
    """Rung 2: linear anchor.

    Standardisation is a convergence requirement for a gradient learner on monetary
    columns spanning six orders of magnitude -- not a tuned hyperparameter. That trees
    need no such step is itself part of the Phase 1 scaling argument.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def tree_depth_tuned(seed=SEED):
    """Rung 3: depth selected by inner CV, so the rung is not handicapped by an
    arbitrary depth. The search is nested inside each outer fold."""
    return GridSearchCV(
        DecisionTreeClassifier(random_state=seed),
        {"max_depth": [2, 3, 4, 5, 6, 8, 10, 12]},
        scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        n_jobs=-1,
    )


def random_forest(seed=SEED):
    """Rung 4: bagging at library defaults, isolating it from boosting."""
    return RandomForestClassifier(random_state=seed, n_jobs=-1)


def ladder_specs(best_params=None):
    """The seven rungs of WP5, in increasing capability and decreasing interpretability.

    Rungs 2 and 3 are untuned internal reference implementations used to locate XGBoost
    on a capability axis. They are NOT the group's Decision Tree or Logistic Regression
    submissions, which are separately tuned by other members.
    """
    specs = [
        {"label": "rung0_majority", "model": "Majority class",
         "factory": majority, "features": RAW_FEATURES,
         "interpretability": "total (1 constant)", "interp_cost": 1},
        {"label": "rung1_pay0_rule", "model": "PAY_0 only, depth-1 tree",
         "factory": single_rule, "features": ["PAY_0"],
         "interpretability": "single rule", "interp_cost": 1},
        {"label": "rung2_logistic", "model": "Logistic regression (defaults)",
         "factory": logistic, "features": RAW_FEATURES,
         "interpretability": "23 coefficients + intercept", "interp_cost": 24},
        {"label": "rung3_tree", "model": "Decision tree (depth-tuned)",
         "factory": tree_depth_tuned, "features": RAW_FEATURES,
         "interpretability": "rule set", "interp_cost": None},
        {"label": "rung4_forest", "model": "Random forest (defaults)",
         "factory": random_forest, "features": RAW_FEATURES,
         "interpretability": "not directly available - see SHAP", "interp_cost": None},
        {"label": "rung5_xgb_default", "model": "XGBoost (library defaults)",
         "factory": xgb_library_defaults, "features": RAW_FEATURES,
         "interpretability": "not directly available - see SHAP", "interp_cost": None},
    ]
    if best_params is not None:
        specs.append({
            "label": "rung6_xgb_tuned", "model": "XGBoost (tuned)",
            "factory": lambda: xgb_classifier(best_params), "features": RAW_FEATURES,
            "interpretability": "not directly available - see SHAP", "interp_cost": None,
        })
    return specs


def scale_pos_weight(y):
    """neg/pos, the ratio XGBoost expects. Confirm polarity: 1 = default."""
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    return float((len(y) - pos) / pos)


def categorical_frame(X, cols=None):
    """Cast PAY_n to pandas `category` for the enable_categorical arm of WP8."""
    out = X.copy()
    for c in (cols if cols is not None else PAY_COLS):
        out[c] = out[c].astype("category")
    return out


class SmoteXGB(BaseEstimator, ClassifierMixin):
    """SMOTE + XGBoost, resampling strictly inside `fit`.

    Equivalent to an imblearn Pipeline for leakage purposes -- oversampling only ever
    sees the rows passed to `fit`, which cv_evaluate restricts to the fold's training
    half -- with one deliberate refinement: the early-stopping slice is carved off
    *before* resampling, so it retains the natural 22% prevalence. Validating against
    synthetically balanced data would make the stopping criterion measure the wrong
    distribution.
    """

    def __init__(self, params=None, seed=SEED):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split

        y = np.asarray(y).astype(int)
        i_fit, i_es = train_test_split(
            np.arange(len(y)), test_size=ES_SLICE, stratify=y, random_state=self.seed
        )
        X_res, y_res = SMOTE(random_state=self.seed).fit_resample(
            X.iloc[i_fit], y[i_fit]
        )
        self.model_ = xgb_classifier(self.params, spw=1.0, seed=self.seed)
        self.model_.fit(X_res, y_res,
                        eval_set=[(X.iloc[i_es], y[i_es])], verbose=False)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def best_iteration(self):
        return self.model_.best_iteration


def count_tree_nodes(fitted):
    """Interpretability cost for the tree rungs: nodes a human must read."""
    est = fitted.best_estimator_ if isinstance(fitted, GridSearchCV) else fitted
    if isinstance(est, DecisionTreeClassifier):
        return int(est.tree_.node_count)
    if isinstance(est, RandomForestClassifier):
        return int(sum(t.tree_.node_count for t in est.estimators_))
    if isinstance(est, XGBClassifier):
        return int(len(est.get_booster().get_dump()))
    return None
