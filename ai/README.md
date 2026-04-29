# Pulsevera AI

Implementation of the **AI Engineer** learning path of Capstone CC26-PRU439 –
*Pulsevera: Predict, Prevent, Prevail*. The module trains a heart-stroke
classification pipeline on the `heart_disease.csv` dataset, explains its
predictions with SHAP, and exposes the model through a FastAPI service that
the Full-Stack Web team can integrate.

## Mapping to the project plan

| Week | Project plan task | Where it lives |
| ---- | ----------------- | -------------- |
| 1 | Understand dataset, set up ML environment | `requirements.txt`, `src/preprocessing.py`, `scripts/explore_data.py` |
| 2 | Train Logistic Regression / Random Forest / Decision Tree | `src/train.py`, `scripts/train_baseline.py` |
| 3 | Hyperparameter tuning, SHAP, inference code | `src/tune.py`, `src/explain.py`, `src/inference.py` |
| 4 | Accuracy / Precision / Recall / F1 evaluation, FastAPI REST API | `src/evaluate.py`, `src/api.py`, `scripts/train_full.py` |
| 5 | Optimization, integration testing, deployment-ready export | `tests/`, `scripts/train_full.py`, `scripts/serve_api.py` |

## Project layout

```
ai/
├── requirements.txt
├── pytest.ini
├── README.md
├── data/                       # place heart_disease.csv here (or keep it at the repo root)
├── models/                     # serialised pipeline + metadata (generated)
├── reports/
│   ├── figures/                # confusion matrix, ROC curve, SHAP summary (generated)
│   └── metrics/                # JSON metrics snapshots (generated)
├── scripts/
│   ├── explore_data.py
│   ├── train_baseline.py
│   ├── train_full.py
│   ├── predict_sample.py
│   └── serve_api.py
├── src/
│   ├── config.py               # paths, columns, target, defaults
│   ├── preprocessing.py        # data loading + ColumnTransformer
│   ├── train.py                # baselines + cross-validation + persistence
│   ├── tune.py                 # GridSearchCV search spaces
│   ├── evaluate.py             # metrics + plots
│   ├── explain.py              # SHAP global & local explanations
│   ├── inference.py            # predict() + recommendations
│   └── api.py                  # FastAPI service
└── tests/                      # preprocessing + inference + API smoke tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# or: source .venv/bin/activate # Linux / macOS

cd ai
pip install -r requirements.txt
```

The dataset can live at either of these locations (the loader tries each in
order):

1. `ai/data/heart_disease.csv`
2. `heart_disease.csv` (project root – this is the current location)

## Workflow

```bash
cd ai

# Week 1 – sanity check the dataset
python -m scripts.explore_data

# Week 2 – cross-validated baselines (no model is saved)
python -m scripts.train_baseline

# Week 3-4 – tune, evaluate, persist, render SHAP summary
python -m scripts.train_full

# Week 4-5 – run a prediction from the CLI
python -m scripts.predict_sample

# Week 4-5 – serve the REST API
python -m scripts.serve_api
# or:
uvicorn src.api:app --reload
```

`scripts/train_full.py` writes the following artifacts:

- `models/pulsevera_pipeline.joblib` – fitted scikit-learn pipeline
- `models/pulsevera_pipeline.meta.json` – selected model, best params, test metrics
- `models/feature_names.json` – names of the post-encoding features
- `reports/figures/<model>_confusion_matrix.png`
- `reports/figures/<model>_roc_curve.png`
- `reports/figures/shap_summary.png`
- `reports/metrics/<model>_metrics.json`
- `reports/metrics/training_summary.json`

## REST API (FastAPI)

Once `scripts/train_full.py` has produced an artifact, start the API and open
`http://localhost:8000/docs` for the interactive Swagger UI.

| Method | Path             | Purpose |
| ------ | ---------------- | ------- |
| GET    | `/health`        | Liveness check + selected model + headline metrics |
| GET    | `/schema`        | Input schema + an example payload |
| POST   | `/predict`       | Single-record risk score, top SHAP factors, recommendations |
| POST   | `/predict/batch` | Same as `/predict` for an array of records |

Example request body for `POST /predict`:

```json
{
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
  "glucose": 105
}
```

Response shape:

```json
{
  "label": "yes",
  "is_high_risk": true,
  "probability": 0.71,
  "risk_score_percent": 71.0,
  "top_risk_factors": [
    {"feature": "sysBP", "impact": 0.18, "direction": "increases"},
    {"feature": "age",   "impact": 0.12, "direction": "increases"},
    {"feature": "BMI",   "impact": 0.08, "direction": "increases"}
  ],
  "recommendations": [
    "Monitor systolic blood pressure and reduce sodium intake.",
    "Routine cardiovascular screening becomes increasingly important with age.",
    "Maintain a healthy BMI through balanced nutrition and regular exercise."
  ],
  "model_name": "random_forest"
}
```

## Testing

```bash
cd ai
pytest
```

The test suite covers:

- Schema + missing-value handling in `src/preprocessing.py`.
- End-to-end inference: it trains a tiny logistic-regression baseline,
  persists it, and validates `predict()` plus the FastAPI endpoints with
  `TestClient`.

## Risks tracked from the project plan

- **Poor data quality** – median / most-frequent imputation lives inside the
  pipeline, so train-time and inference-time handling stay identical.
- **Insufficient model performance** – three algorithms are benchmarked with
  stratified K-fold CV and Grid Search before the best is persisted.
- **Lack of interpretability** – SHAP global summary plots and local Top-3
  factors per prediction are produced.
- **Integration failure** – FastAPI app + `TestClient` smoke tests guard the
  contract used by the Full-Stack team.
- **Overfitting** – stratified CV + held-out test set; class weights are
  balanced to handle the imbalanced target.
