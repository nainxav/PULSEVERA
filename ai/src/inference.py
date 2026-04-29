"""Inference utilities for Pulsevera (Week 3-5 of the AI path).

Loads the persisted pipeline lazily, validates input rows, and produces a
prediction object that the FastAPI service can return directly. Includes
optional SHAP-driven local explanations to surface the top risk factors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config
from .explain import compute_shap_values, top_local_factors
from .preprocessing import basic_clean, load_raw_dataset, split_features_target


@dataclass
class RiskFactor:
    feature: str
    impact: float
    direction: str


@dataclass
class PredictionResult:
    label: str
    is_high_risk: bool
    probability: float
    risk_score_percent: float
    top_risk_factors: list[RiskFactor] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    model_name: str = ""


_RECOMMENDATION_LIBRARY: dict[str, str] = {
    "age": "Routine cardiovascular screening becomes increasingly important with age.",
    "BMI": "Maintain a healthy BMI through balanced nutrition and regular exercise.",
    "sysBP": "Monitor systolic blood pressure and reduce sodium intake.",
    "diaBP": "Track diastolic blood pressure; manage stress and sleep quality.",
    "totChol": "Reduce saturated fats and consider regular cholesterol checks.",
    "glucose": "Limit refined sugars and monitor fasting glucose periodically.",
    "heartRate": "Engage in moderate cardio to keep resting heart rate in a healthy range.",
    "cigsPerDay": "Quit smoking; even cutting down meaningfully reduces risk.",
    "currentSmoker": "Quit smoking; even cutting down meaningfully reduces risk.",
    "BPMeds": "Adhere strictly to prescribed blood pressure medication.",
    "prevalentHyp": "Follow a hypertension-friendly lifestyle (DASH-style diet).",
    "diabetes": "Control blood sugar through diet, exercise, and medication adherence.",
    "prevalentStroke": "Discuss secondary prevention strategies with a cardiologist.",
}


@lru_cache(maxsize=1)
def load_pipeline(model_path: str | None = None) -> Pipeline:
    """Load the serialised pipeline (cached for the process lifetime)."""

    path = Path(model_path) if model_path else config.MODELS_DIR / config.ARTIFACTS.pipeline
    if not path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found at {path}. Run "
            "`python -m ai.scripts.train_full` first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata() -> Mapping[str, Any]:
    """Read the metadata JSON saved alongside the pipeline."""

    path = config.MODELS_DIR / config.ARTIFACTS.metadata
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _background_sample() -> pd.DataFrame:
    """Cache a small background sample for SHAP explainers."""

    raw = basic_clean(load_raw_dataset())
    features, _ = split_features_target(raw)
    return features.sample(n=min(200, len(features)), random_state=config.RANDOM_STATE)


def _to_dataframe(payload: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalise dict / list-of-dicts payloads into a DataFrame."""

    if isinstance(payload, Mapping):
        rows = [dict(payload)]
    else:
        rows = [dict(row) for row in payload]
    frame = pd.DataFrame(rows)

    missing = [c for c in config.ALL_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    return basic_clean(frame[list(config.ALL_FEATURES)])


def _build_recommendations(
    factors: Iterable[RiskFactor], max_recommendations: int = 3
) -> list[str]:
    """Pick lifestyle recommendations matching the dominant SHAP features."""

    seen: set[str] = set()
    recommendations: list[str] = []
    for factor in factors:
        base_feature = factor.feature.split("_")[0]
        for key, advice in _RECOMMENDATION_LIBRARY.items():
            if key.lower() in factor.feature.lower() or key == base_feature:
                if advice not in seen:
                    recommendations.append(advice)
                    seen.add(advice)
                break
        if len(recommendations) >= max_recommendations:
            break
    return recommendations


def predict(
    payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    explain: bool = True,
    top_n_factors: int = 3,
) -> list[PredictionResult]:
    """Run prediction (+ optional SHAP explanation) on one or many records."""

    pipeline = load_pipeline()
    metadata = load_metadata()
    frame = _to_dataframe(payload)

    probabilities = pipeline.predict_proba(frame)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    explanation = None
    if explain:
        explanation = compute_shap_values(
            pipeline=pipeline,
            background=_background_sample(),
            samples=frame,
        )

    results: list[PredictionResult] = []
    for idx, (probability, label) in enumerate(zip(probabilities, predictions)):
        risk_factors: list[RiskFactor] = []
        if explanation is not None:
            for contribution in top_local_factors(
                explanation, row_index=idx, top_n=top_n_factors
            ):
                risk_factors.append(
                    RiskFactor(
                        feature=contribution.feature,
                        impact=round(contribution.shap_value, 4),
                        direction=contribution.direction,
                    )
                )
        results.append(
            PredictionResult(
                label=config.POSITIVE_LABEL if label == 1 else config.NEGATIVE_LABEL,
                is_high_risk=bool(label),
                probability=float(probability),
                risk_score_percent=round(float(probability) * 100, 2),
                top_risk_factors=risk_factors,
                recommendations=_build_recommendations(risk_factors),
                model_name=str(metadata.get("model_name", "unknown")),
            )
        )
    return results


def predict_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convenience wrapper used by the API: take/return plain dicts."""

    [result] = predict(payload, explain=True)
    output = asdict(result)
    output["top_risk_factors"] = [asdict(f) for f in result.top_risk_factors]
    return output
