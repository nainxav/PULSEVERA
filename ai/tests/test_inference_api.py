"""Smoke tests that train a tiny model and exercise the inference + API.

The training step here uses a small subset and only the Logistic Regression
candidate so the suite stays fast (≈ a few seconds). It is sufficient to prove
that the end-to-end wiring (pipeline → joblib artifact → FastAPI) is healthy.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src import config
from src.api import create_app
from src.evaluate import evaluate_model
from src.inference import load_metadata, load_pipeline, predict
from src.preprocessing import (
    basic_clean,
    load_raw_dataset,
    split_features_target,
)
from src.train import TrainedModel, build_candidate_models, make_pipeline, save_model


SAMPLE_PROFILE = {
    "Gender": "Male",
    "age": 58,
    "education": "graduate",
    "currentSmoker": 1,
    "cigsPerDay": 20,
    "BPMeds": 0,
    "prevalentStroke": "no",
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 245,
    "sysBP": 152,
    "diaBP": 95,
    "BMI": 29.4,
    "heartRate": 84,
    "glucose": 105,
}


@pytest.fixture(scope="module", autouse=True)
def trained_artifact():
    """Train a quick logistic-regression baseline and persist it once."""

    config.ensure_directories()
    raw = basic_clean(load_raw_dataset())
    features, target = split_features_target(raw)

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=config.RANDOM_STATE
    )

    pipeline = make_pipeline(build_candidate_models()["logistic_regression"])
    pipeline.fit(features_train, target_train)

    report = evaluate_model(
        "logistic_regression", pipeline, features_test, target_test
    )

    save_model(
        TrainedModel(
            name="logistic_regression",
            pipeline=pipeline,
            cv_scores={"f1": report.f1},
        ),
        extra_metadata={"test_metrics": {"accuracy": report.accuracy, "f1": report.f1}},
    )

    load_pipeline.cache_clear()
    load_metadata.cache_clear()
    yield


def test_predict_returns_probability_in_unit_interval() -> None:
    [result] = predict(SAMPLE_PROFILE, explain=False)
    assert 0.0 <= result.probability <= 1.0
    assert result.label in (config.POSITIVE_LABEL, config.NEGATIVE_LABEL)


def test_predict_with_explanations_returns_top_factors() -> None:
    [result] = predict(SAMPLE_PROFILE, explain=True, top_n_factors=3)
    assert len(result.top_risk_factors) == 3
    for factor in result.top_risk_factors:
        assert factor.direction in {"increases", "decreases"}


def test_api_health_endpoint_returns_metadata() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"]


def test_api_predict_endpoint_returns_serialisable_payload() -> None:
    client = TestClient(create_app())
    response = client.post("/predict", json=SAMPLE_PROFILE)
    assert response.status_code == 200, response.text

    body = response.json()
    json.dumps(body)  # must be JSON-serialisable end-to-end
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in (config.POSITIVE_LABEL, config.NEGATIVE_LABEL)
    assert isinstance(body["recommendations"], list)


def test_api_rejects_invalid_input() -> None:
    client = TestClient(create_app())
    bad_payload = dict(SAMPLE_PROFILE)
    bad_payload["age"] = 5  # below the validator's lower bound
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
