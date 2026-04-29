"""Project-wide configuration for Pulsevera AI path.

Defines paths, dataset schema, target column, and modelling defaults so that
preprocessing, training, evaluation, explanation, and the inference API all
share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
AI_DIR: Path = ROOT_DIR / "ai"

DATA_DIR: Path = AI_DIR / "data"
MODELS_DIR: Path = AI_DIR / "models"
REPORTS_DIR: Path = AI_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

RAW_DATASET_CANDIDATES: tuple[Path, ...] = (
    DATA_DIR / "heart_disease.csv",
    ROOT_DIR / "heart_disease.csv",
)

TARGET_COLUMN: str = "Heart_ stroke"
TARGET_RENAMED: str = "heart_stroke"

NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
)

BINARY_FEATURES: tuple[str, ...] = (
    "currentSmoker",
    "BPMeds",
    "prevalentHyp",
    "diabetes",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Gender",
    "education",
    "prevalentStroke",
)

ALL_FEATURES: tuple[str, ...] = (
    NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
)

POSITIVE_LABEL: str = "yes"
NEGATIVE_LABEL: str = "No"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5


@dataclass(frozen=True)
class ModelArtifact:
    """Filenames for serialised model artifacts."""

    pipeline: str = "pulsevera_pipeline.joblib"
    metadata: str = "pulsevera_pipeline.meta.json"
    feature_names: str = "feature_names.json"


ARTIFACTS: ModelArtifact = ModelArtifact()


@dataclass(frozen=True)
class TrainingDefaults:
    """Defaults shared by baseline training and tuning."""

    scoring: str = "f1"
    n_jobs: int = -1
    class_weight: str = "balanced"
    cv_folds: int = CV_FOLDS
    random_state: int = RANDOM_STATE
    candidate_models: tuple[str, ...] = field(
        default_factory=lambda: ("logistic_regression", "random_forest", "decision_tree")
    )


TRAINING: TrainingDefaults = TrainingDefaults()


def ensure_directories() -> None:
    """Create writable working directories on first run."""

    for directory in (DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path() -> Path:
    """Return the first existing dataset path from the candidates."""

    for candidate in RAW_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "heart_disease.csv was not found. Place it in ai/data/ "
        "or in the project root."
    )
