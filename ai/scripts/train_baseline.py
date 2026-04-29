"""Week 2 baseline trainer.

Cross-validates the three required classifiers (Logistic Regression, Random
Forest, Decision Tree) and prints a comparative summary. Does NOT persist a
final model — use ``train_full.py`` for the production artifact.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.preprocessing import (
    basic_clean,
    load_raw_dataset,
    split_features_target,
)
from src.train import cross_validate_models


def _format(scores: dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.3f}" for k, v in scores.items())


def main() -> None:
    config.ensure_directories()

    raw = basic_clean(load_raw_dataset())
    features, target = split_features_target(raw)

    features_train, _, target_train, _ = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        stratify=target,
        random_state=config.RANDOM_STATE,
    )

    cv_scores = cross_validate_models(features_train, target_train)
    print("Baseline cross-validation (5-fold, stratified)")
    print("----------------------------------------------")
    for name, scores in cv_scores.items():
        print(f"{name:>22} | {_format(scores)}")

    output = config.METRICS_DIR / "baseline_cv_scores.json"
    output.write_text(json.dumps(cv_scores, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
