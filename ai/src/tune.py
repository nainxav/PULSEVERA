"""Hyperparameter tuning for Pulsevera baseline classifiers.

Implements Week 3 of the AI path: model selection + tuning. The search spaces
are intentionally compact so they finish in a few minutes on a laptop while
still demonstrating the effect of regularisation, tree depth, and ensemble
size on the F1 score for the heart-stroke target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config
from .train import build_candidate_models, make_pipeline


SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 6, 10, 16],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "decision_tree": {
        "classifier__max_depth": [None, 4, 6, 8, 12],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__criterion": ["gini", "entropy"],
    },
}


@dataclass
class TuningResult:
    """Output of a single GridSearchCV run."""

    name: str
    best_params: dict[str, Any]
    best_score: float
    estimator: Any


def tune_model(
    name: str,
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str = "f1",
    cv_folds: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> TuningResult:
    """Run GridSearchCV on a single named candidate."""

    if name not in SEARCH_SPACES:
        raise KeyError(f"No search space for '{name}'. Known: {list(SEARCH_SPACES)}.")

    estimator = build_candidate_models(random_state)[name]
    pipeline = make_pipeline(estimator)
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=SEARCH_SPACES[name],
        scoring=scoring,
        cv=folds,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(features, target)

    return TuningResult(
        name=name,
        best_params={k: _to_jsonable(v) for k, v in search.best_params_.items()},
        best_score=float(search.best_score_),
        estimator=search.best_estimator_,
    )


def tune_all(
    features: pd.DataFrame,
    target: pd.Series,
    scoring: str = "f1",
    cv_folds: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, TuningResult]:
    """Tune every model in :data:`SEARCH_SPACES`."""

    results: dict[str, TuningResult] = {}
    for name in SEARCH_SPACES:
        results[name] = tune_model(
            name=name,
            features=features,
            target=target,
            scoring=scoring,
            cv_folds=cv_folds,
            random_state=random_state,
        )
    return results


def best_of(results: dict[str, TuningResult]) -> TuningResult:
    """Return the tuning result with the highest ``best_score``."""

    return max(results.values(), key=lambda r: r.best_score)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars / None into plain JSON-serialisable values."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
