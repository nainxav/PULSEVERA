"""End-to-end Week 3-4 trainer.

Pipeline:
    1. Load + clean dataset.
    2. Train/test split (stratified, fixed seed).
    3. Tune Logistic Regression / Random Forest / Decision Tree.
    4. Pick the best by F1 and refit on the full training split.
    5. Evaluate on the held-out test set (Accuracy / Precision / Recall / F1).
    6. Persist the pipeline + metadata + SHAP global summary plot.
"""

from __future__ import annotations

import json

from sklearn.model_selection import train_test_split

from src import config
from src.evaluate import evaluate_and_persist
from src.explain import compute_shap_values, save_summary_plot
from src.preprocessing import (
    basic_clean,
    load_raw_dataset,
    split_features_target,
)
from src.train import TrainedModel, save_model
from src.tune import best_of, tune_all


def main() -> None:
    config.ensure_directories()

    raw = basic_clean(load_raw_dataset())
    features, target = split_features_target(raw)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        stratify=target,
        random_state=config.RANDOM_STATE,
    )

    print("Tuning candidate models with GridSearchCV (5-fold, scoring=f1)...")
    tuning_results = tune_all(features_train, target_train, scoring="f1")
    for name, result in tuning_results.items():
        print(f"  {name:>22} | best F1 (CV) = {result.best_score:.3f} | params = {result.best_params}")

    best = best_of(tuning_results)
    print(f"\nBest tuned model: {best.name} (CV F1 = {best.best_score:.3f})")

    print("\nEvaluating on held-out test set...")
    report = evaluate_and_persist(
        name=best.name,
        pipeline=best.estimator,
        features=features_test,
        target=target_test,
    )
    print(
        f"  accuracy={report.accuracy:.3f} | precision={report.precision:.3f} | "
        f"recall={report.recall:.3f} | f1={report.f1:.3f} | "
        f"roc_auc={report.roc_auc if report.roc_auc is None else f'{report.roc_auc:.3f}'}"
    )

    print("\nComputing global SHAP summary plot...")
    background = features_train.sample(
        n=min(200, len(features_train)), random_state=config.RANDOM_STATE
    )
    shap_sample = features_test.sample(
        n=min(300, len(features_test)), random_state=config.RANDOM_STATE
    )
    explanation = compute_shap_values(best.estimator, background, shap_sample)
    summary_path = save_summary_plot(explanation)
    print(f"  saved: {summary_path}")

    print("\nPersisting final model artifacts...")
    trained = TrainedModel(
        name=best.name,
        pipeline=best.estimator,
        cv_scores={"f1": best.best_score},
    )
    artifact_path = save_model(
        trained,
        extra_metadata={
            "best_params": best.best_params,
            "test_metrics": {
                "accuracy": report.accuracy,
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "roc_auc": report.roc_auc,
            },
            "tuning_results": {
                name: {
                    "best_score": result.best_score,
                    "best_params": result.best_params,
                }
                for name, result in tuning_results.items()
            },
        },
    )
    print(f"  saved: {artifact_path}")

    summary_payload = {
        "selected_model": best.name,
        "cv_f1": best.best_score,
        "test_metrics": {
            "accuracy": report.accuracy,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "roc_auc": report.roc_auc,
        },
    }
    summary_file = config.METRICS_DIR / "training_summary.json"
    summary_file.write_text(json.dumps(summary_payload, indent=2))
    print(f"  saved: {summary_file}")


if __name__ == "__main__":
    main()
