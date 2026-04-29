"""Data loading & preprocessing for the Pulsevera heart-disease dataset.

Implements Week 1 of the AI learning path: understand the dataset from a
modelling perspective and build a reusable preprocessing pipeline that handles
missing values, scales numeric features, and encodes categorical features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass(frozen=True)
class DatasetSummary:
    """Lightweight description of the dataset for reporting."""

    n_rows: int
    n_features: int
    target_distribution: dict[str, int]
    missing_values: dict[str, int]


def load_raw_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the raw CSV, normalising the target column name."""

    csv_path = config.resolve_dataset_path() if path is None else path
    frame = pd.read_csv(csv_path, na_values=["NA", "na", "N/A", ""])

    if config.TARGET_COLUMN in frame.columns:
        frame = frame.rename(columns={config.TARGET_COLUMN: config.TARGET_RENAMED})

    frame.columns = [c.strip() for c in frame.columns]
    return frame


def summarise(frame: pd.DataFrame) -> DatasetSummary:
    """Produce a quick descriptive summary used by reports & sanity checks."""

    target = frame[config.TARGET_RENAMED]
    distribution = target.value_counts(dropna=False).to_dict()
    missing = frame.isna().sum()
    missing = missing[missing > 0].to_dict()
    return DatasetSummary(
        n_rows=len(frame),
        n_features=frame.shape[1] - 1,
        target_distribution={str(k): int(v) for k, v in distribution.items()},
        missing_values={str(k): int(v) for k, v in missing.items()},
    )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with a binary 0/1 integer target."""

    feature_columns = list(config.ALL_FEATURES)
    missing_columns = [c for c in feature_columns if c not in frame.columns]
    if missing_columns:
        raise KeyError(f"Dataset is missing required columns: {missing_columns}")

    features = frame[feature_columns].copy()
    target_raw = frame[config.TARGET_RENAMED].astype(str).str.strip().str.lower()
    target = (target_raw == config.POSITIVE_LABEL.lower()).astype(int)
    target.name = config.TARGET_RENAMED
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing ``ColumnTransformer``.

    - Numeric columns: median imputation + standardisation.
    - Binary columns: most-frequent imputation (kept as 0/1).
    - Categorical columns: most-frequent imputation + one-hot encoding.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(config.NUMERIC_FEATURES)),
            ("bin", binary_pipeline, list(config.BINARY_FEATURES)),
            ("cat", categorical_pipeline, list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-transform feature names of a fitted preprocessor."""

    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps[
            "encoder"
        ]
        cat_names = list(encoder.get_feature_names_out(config.CATEGORICAL_FEATURES))
        return list(config.NUMERIC_FEATURES) + list(config.BINARY_FEATURES) + cat_names


def basic_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Light-touch cleaning: trim whitespace and normalise text values.

    The bulk of imputation happens inside the pipeline, which keeps train and
    inference behaviour identical.
    """

    frame = frame.copy()
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan})
