"""Model training utilities for the Pulsevera AI path.

Implements the Week 2 requirement of training three classification baselines
(Logistic Regression, Random Forest, Decision Tree) and the Week 3 requirement
of selecting the best one via cross-validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import config
from .preprocessing import (
    build_preprocessor,
    get_feature_names,
    load_raw_dataset,
    basic_clean,
    split_features_target,
)


SCORING: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


@dataclass
class TrainedModel:
    """Container for a trained pipeline and its cross-validated scores."""

    name: str
    pipeline: Pipeline
    cv_scores: dict[str, float]


def build_candidate_models(random_state: int = config.RANDOM_STATE) -> dict[str, ClassifierMixin]:
    """Return the dictionary of baseline classifiers required by the plan."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def make_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Wrap ``estimator`` with the shared preprocessing transformer."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def cross_validate_models(
    features: pd.DataFrame,
    target: pd.Series,
    models: Mapping[str, ClassifierMixin] | None = None,
    cv_folds: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Run stratified k-fold CV for each candidate model.

    Returns a mapping of ``model_name -> {metric: mean_score}``.
    """

    candidates = models or build_candidate_models(random_state)
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        cv = cross_validate(
            pipeline,
            features,
            target,
            cv=folds,
            scoring=list(SCORING),
            n_jobs=-1,
            return_train_score=False,
        )
        results[name] = {
            metric: float(np.mean(cv[f"test_{metric}"])) for metric in SCORING
        }
    return results


def train_baselines(
    features: pd.DataFrame,
    target: pd.Series,
    cv_folds: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, TrainedModel]:
    """Fit each baseline on the full training set after evaluating with CV."""

    cv_scores = cross_validate_models(
        features, target, cv_folds=cv_folds, random_state=random_state
    )

    trained: dict[str, TrainedModel] = {}
    for name, estimator in build_candidate_models(random_state).items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(features, target)
        trained[name] = TrainedModel(
            name=name, pipeline=pipeline, cv_scores=cv_scores[name]
        )
    return trained


def select_best(trained: Mapping[str, TrainedModel], metric: str = "f1") -> TrainedModel:
    """Pick the trained model with the highest mean ``metric``."""

    if metric not in SCORING:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {SCORING}.")
    return max(trained.values(), key=lambda m: m.cv_scores[metric])


def save_model(
    trained: TrainedModel,
    models_dir: Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the pipeline + a JSON sidecar with metadata for inference."""

    config.ensure_directories()
    target_dir = models_dir or config.MODELS_DIR
    pipeline_path = target_dir / config.ARTIFACTS.pipeline
    metadata_path = target_dir / config.ARTIFACTS.metadata
    feature_names_path = target_dir / config.ARTIFACTS.feature_names

    joblib.dump(trained.pipeline, pipeline_path)

    feature_names = get_feature_names(trained.pipeline.named_steps["preprocessor"])
    feature_names_path.write_text(json.dumps(feature_names, indent=2))

    metadata: dict[str, Any] = {
        "model_name": trained.name,
        "cv_scores": trained.cv_scores,
        "input_columns": list(config.ALL_FEATURES),
        "target": config.TARGET_RENAMED,
        "positive_label": config.POSITIVE_LABEL,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return pipeline_path


def train_and_persist(
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> TrainedModel:
    """End-to-end Week-2 routine: load → CV → fit baselines → keep the best."""

    raw = basic_clean(load_raw_dataset())
    features, target = split_features_target(raw)

    features_train, _, target_train, _ = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    trained = train_baselines(features_train, target_train, random_state=random_state)
    best = select_best(trained, metric="f1")
    save_model(best, extra_metadata={"selection_metric": "f1"})
    return best
