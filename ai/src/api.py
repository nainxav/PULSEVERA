"""FastAPI service for Pulsevera heart-stroke risk prediction.

Implements Week 4 of the AI path: expose the trained model behind a REST API
that the full-stack web team can consume. Endpoints:

- ``GET  /health`` — liveness + model metadata
- ``GET  /schema`` — input schema with example payload
- ``POST /predict`` — single-record prediction with risk score, top factors,
  and lifestyle recommendations
- ``POST /predict/batch`` — batched prediction for analytics dashboards
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from . import config
from .inference import (
    PredictionResult,
    load_metadata,
    load_pipeline,
    predict,
)


class HealthInput(BaseModel):
    """Request payload describing one user's health profile."""

    model_config = ConfigDict(extra="forbid")

    Gender: Literal["Male", "Female"] = Field(..., description="Biological sex")
    age: int = Field(..., ge=10, le=120)
    education: Literal["uneducated", "primaryschool", "graduate", "postgraduate"]
    currentSmoker: Literal[0, 1]
    cigsPerDay: float = Field(..., ge=0, le=80)
    BPMeds: Literal[0, 1] | None = Field(
        default=0, description="Currently on blood pressure medication"
    )
    prevalentStroke: Literal["yes", "no"] = Field(default="no")
    prevalentHyp: Literal[0, 1] = Field(default=0)
    diabetes: Literal[0, 1] = Field(default=0)
    totChol: float = Field(..., ge=80, le=700, description="Total cholesterol (mg/dL)")
    sysBP: float = Field(..., ge=70, le=300, description="Systolic blood pressure")
    diaBP: float = Field(..., ge=40, le=200, description="Diastolic blood pressure")
    BMI: float = Field(..., ge=10, le=70, description="Body mass index")
    heartRate: float = Field(..., ge=30, le=220)
    glucose: float = Field(..., ge=30, le=600)


class BatchInput(BaseModel):
    items: list[HealthInput]


class RiskFactorOut(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionOut(BaseModel):
    label: str
    is_high_risk: bool
    probability: float
    risk_score_percent: float
    top_risk_factors: list[RiskFactorOut]
    recommendations: list[str]
    model_name: str


class HealthCheck(BaseModel):
    status: str
    model_name: str
    metrics: dict[str, Any]


def _serialise(result: PredictionResult) -> PredictionOut:
    return PredictionOut(
        label=result.label,
        is_high_risk=result.is_high_risk,
        probability=result.probability,
        risk_score_percent=result.risk_score_percent,
        top_risk_factors=[RiskFactorOut(**f.__dict__) for f in result.top_risk_factors],
        recommendations=result.recommendations,
        model_name=result.model_name,
    )


def create_app() -> FastAPI:
    """Application factory used by uvicorn and tests."""

    app = FastAPI(
        title="Pulsevera AI – Heart Stroke Risk API",
        version="0.1.0",
        description=(
            "REST API serving the Pulsevera classification pipeline. Returns "
            "a calibrated risk score plus the top SHAP factors and lifestyle "
            "recommendations for the front-end."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _warm_up() -> None:
        try:
            load_pipeline()
        except FileNotFoundError:
            # Allow the API to start without a model; /health will surface it.
            pass

    @app.get("/health", response_model=HealthCheck)
    def health() -> HealthCheck:
        try:
            load_pipeline()
            metadata = dict(load_metadata())
            return HealthCheck(
                status="ok",
                model_name=str(metadata.get("model_name", "unknown")),
                metrics=dict(metadata.get("test_metrics", metadata.get("cv_scores", {}))),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        example = {
            "Gender": "Male",
            "age": 52,
            "education": "graduate",
            "currentSmoker": 1,
            "cigsPerDay": 10,
            "BPMeds": 0,
            "prevalentStroke": "no",
            "prevalentHyp": 1,
            "diabetes": 0,
            "totChol": 245,
            "sysBP": 138,
            "diaBP": 88,
            "BMI": 27.4,
            "heartRate": 78,
            "glucose": 92,
        }
        return {
            "input_columns": list(config.ALL_FEATURES),
            "target": config.TARGET_RENAMED,
            "positive_label": config.POSITIVE_LABEL,
            "example": example,
        }

    @app.post("/predict", response_model=PredictionOut)
    def predict_one(payload: HealthInput) -> PredictionOut:
        try:
            [result] = predict(payload.model_dump(), explain=True)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _serialise(result)

    @app.post("/predict/batch", response_model=list[PredictionOut])
    def predict_batch(payload: BatchInput) -> list[PredictionOut]:
        if not payload.items:
            raise HTTPException(status_code=422, detail="`items` must be non-empty.")
        try:
            results = predict(
                [item.model_dump() for item in payload.items], explain=True
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return [_serialise(r) for r in results]

    return app


app = create_app()
