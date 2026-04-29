"""Unit tests for the preprocessing pipeline.

These tests do not require the trained model artifact; they only exercise the
data layer to keep CI fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import (
    basic_clean,
    build_preprocessor,
    load_raw_dataset,
    split_features_target,
    summarise,
)


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    return basic_clean(load_raw_dataset())


def test_dataset_loads_with_expected_columns(raw_frame: pd.DataFrame) -> None:
    assert config.TARGET_RENAMED in raw_frame.columns
    for column in config.ALL_FEATURES:
        assert column in raw_frame.columns


def test_target_is_binary(raw_frame: pd.DataFrame) -> None:
    _, target = split_features_target(raw_frame)
    assert set(target.unique()) <= {0, 1}
    assert target.sum() > 0  # at least one positive case


def test_summary_counts_match_frame(raw_frame: pd.DataFrame) -> None:
    summary = summarise(raw_frame)
    assert summary.n_rows == len(raw_frame)
    assert summary.n_features == raw_frame.shape[1] - 1


def test_preprocessor_handles_missing_values(raw_frame: pd.DataFrame) -> None:
    features, target = split_features_target(raw_frame)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features, target)

    assert transformed.shape[0] == len(features)
    assert not np.isnan(np.asarray(transformed)).any()
