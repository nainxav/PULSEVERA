"""Model evaluation utilities (Week 3-4 of the AI path).

Computes the metrics required by the project plan (Accuracy, Precision, Recall,
F1) plus a confusion matrix and ROC-AUC. Saves both numeric metrics (JSON) and
visual reports (PNG) under ``ai/reports/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering for CI / scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from . import config


@dataclass
class EvaluationReport:
    """Structured evaluation result for a single model."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> EvaluationReport:
    """Run a full evaluation on a held-out test set."""

    predictions = pipeline.predict(features)

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)[:, 1]
        roc_auc = float(roc_auc_score(target, proba))
    else:
        roc_auc = None

    return EvaluationReport(
        name=name,
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(target, predictions).tolist(),
        classification_report=classification_report(
            target, predictions, output_dict=True, zero_division=0
        ),
    )


def save_metrics(report: EvaluationReport, metrics_dir: Path | None = None) -> Path:
    """Persist the metrics JSON next to the trained pipeline."""

    config.ensure_directories()
    target_dir = metrics_dir or config.METRICS_DIR
    path = target_dir / f"{report.name}_metrics.json"
    payload = {
        "model_name": report.name,
        "accuracy": report.accuracy,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "roc_auc": report.roc_auc,
        "confusion_matrix": report.confusion_matrix,
        "classification_report": report.classification_report,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_confusion_matrix(
    report: EvaluationReport, figures_dir: Path | None = None
) -> Path:
    """Render the confusion matrix to ``ai/reports/figures``."""

    config.ensure_directories()
    target_dir = figures_dir or config.FIGURES_DIR
    path = target_dir / f"{report.name}_confusion_matrix.png"

    matrix = np.array(report.confusion_matrix)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix – {report.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_roc_curve(
    name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    figures_dir: Path | None = None,
) -> Path | None:
    """Render the ROC curve when the model exposes ``predict_proba``."""

    if not hasattr(pipeline, "predict_proba"):
        return None

    config.ensure_directories()
    target_dir = figures_dir or config.FIGURES_DIR
    path = target_dir / f"{name}_roc_curve.png"

    proba = pipeline.predict_proba(features)[:, 1]
    fpr, tpr, _ = roc_curve(target, proba)
    auc = roc_auc_score(target, proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve – {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_and_persist(
    name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> EvaluationReport:
    """Evaluate, save metrics JSON, plus confusion-matrix + ROC plots."""

    report = evaluate_model(name=name, pipeline=pipeline, features=features, target=target)
    save_metrics(report)
    save_confusion_matrix(report)
    save_roc_curve(name, pipeline, features, target)
    return report
