"""SHAP-based interpretability for Pulsevera (Week 3 of the AI path).

Provides helpers to compute SHAP values for a fitted scikit-learn pipeline and
to render the global "summary" plot plus a local explanation for a single
prediction. The local explanation is also used by the inference layer to power
the "Top-3 risk factors" UX promised in the project plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import get_feature_names


@dataclass
class LocalExplanation:
    """Top-N SHAP contributions for a single prediction."""

    feature: str
    value: float
    shap_value: float
    direction: str  # "increases" | "decreases"


def _transform_dataframe(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessor only and return a DataFrame with column names."""

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(frame)
    feature_names = get_feature_names(preprocessor)
    return pd.DataFrame(transformed, columns=feature_names, index=frame.index)


def _build_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    """Pick the cheapest SHAP explainer that fits the underlying model."""

    classifier = pipeline.named_steps["classifier"]
    transformed_background = _transform_dataframe(pipeline, background)

    if hasattr(classifier, "estimators_") or hasattr(classifier, "tree_"):
        return shap.TreeExplainer(classifier, transformed_background)

    masker = shap.maskers.Independent(transformed_background, max_samples=200)
    return shap.LinearExplainer(classifier, masker)


def compute_shap_values(
    pipeline: Pipeline,
    background: pd.DataFrame,
    samples: pd.DataFrame,
) -> shap.Explanation:
    """Return a SHAP ``Explanation`` for the positive class."""

    explainer = _build_explainer(pipeline, background)
    transformed = _transform_dataframe(pipeline, samples)
    explanation = explainer(transformed)

    # For binary classifiers TreeExplainer returns shape (n, n_features, 2);
    # collapse onto the positive-class slice for downstream consumers.
    if explanation.values.ndim == 3:
        explanation = shap.Explanation(
            values=explanation.values[..., 1],
            base_values=np.atleast_1d(explanation.base_values[..., 1]),
            data=explanation.data,
            feature_names=explanation.feature_names,
        )
    return explanation


def save_summary_plot(
    explanation: shap.Explanation, figures_dir: Path | None = None
) -> Path:
    """Persist the global SHAP summary (bar) plot."""

    config.ensure_directories()
    target_dir = figures_dir or config.FIGURES_DIR
    path = target_dir / "shap_summary.png"

    plt.figure(figsize=(8, 6))
    shap.plots.bar(explanation, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def top_local_factors(
    explanation: shap.Explanation,
    row_index: int = 0,
    top_n: int = 3,
) -> list[LocalExplanation]:
    """Return the ``top_n`` SHAP contributors for a single sample."""

    if row_index >= len(explanation):
        raise IndexError(f"row_index {row_index} out of range for explanation")

    values = np.asarray(explanation.values[row_index]).ravel()
    data = np.asarray(explanation.data[row_index]).ravel()
    feature_names: Iterable[str] = explanation.feature_names

    order = np.argsort(np.abs(values))[::-1][:top_n]
    contributions: list[LocalExplanation] = []
    for idx in order:
        contributions.append(
            LocalExplanation(
                feature=str(list(feature_names)[idx]),
                value=float(data[idx]),
                shap_value=float(values[idx]),
                direction="increases" if values[idx] >= 0 else "decreases",
            )
        )
    return contributions
