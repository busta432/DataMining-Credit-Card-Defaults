"""Optuna TPE search.

ROC-AUC is the objective: it is the declared primary metric (Lessmann et al., 2015),
prevalence-independent so it stays comparable across the complexity ladder, and
lower-variance across folds than PR-AUC. `aucpr` is logged every trial but not optimised.

`scale_pos_weight` is deliberately absent from the space. ROC-AUC is invariant to
monotone score transformations, so the tuner would be blind to it and pick arbitrarily;
fixing it per arm turns the imbalance study into a controlled comparison instead.
"""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import N_TRIALS, SEARCH_SPACE, SEED
from .models import xgb_classifier
from .runner import fit_one

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial, space=None):
    space = space or SEARCH_SPACE
    params = {}
    for name, (kind, lo, hi, log) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=log)
    return params


def make_objective(X, y, folds, spw=1.0, seed=SEED, space=None, model_fn=None):
    """`model_fn(params, spw, seed) -> estimator` lets each imbalance arm be tuned
    independently while sharing one search space and one objective."""
    y = np.asarray(y).astype(int)
    if model_fn is None:
        def model_fn(params, spw, seed):
            return xgb_classifier(params, spw=spw, seed=seed)

    def objective(trial):
        params = suggest(trial, space)
        aucs, aps, iters = [], [], []
        for k, (tr, va) in enumerate(folds):
            est = fit_one(model_fn(params, spw, seed), X.iloc[tr], y[tr], seed=seed)
            proba = est.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y[va], proba))
            aps.append(average_precision_score(y[va], proba))
            if getattr(est, "best_iteration", None) is not None:
                iters.append(int(est.best_iteration))
            # Report after each fold so MedianPruner can abandon hopeless trials early.
            trial.report(float(np.mean(aucs)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("pr_auc_mean", float(np.mean(aps)))
        trial.set_user_attr("roc_auc_sd", float(np.std(aucs, ddof=1)))
        trial.set_user_attr("best_iteration_mean", float(np.mean(iters)) if iters else None)
        return float(np.mean(aucs))

    return objective


def run_study(X, y, folds, storage_path, study_name="xgb_roc_auc", n_trials=N_TRIALS,
              spw=1.0, seed=SEED, space=None, model_fn=None):
    """TPE with median pruning, persisted to SQLite so the study resumes and the
    optimisation-history and param-importance plots survive for the appendix."""
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        load_if_exists=True,
    )
    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])
    remaining = max(0, n_trials - done)
    if remaining:
        # XGBoost already parallelises across threads; Optuna stays single-process to
        # avoid thread oversubscription.
        study.optimize(make_objective(X, y, folds, spw=spw, seed=seed, space=space,
                                      model_fn=model_fn),
                       n_trials=remaining, n_jobs=1, show_progress_bar=False)
    return study
