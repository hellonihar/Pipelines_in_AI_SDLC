# Level 2: Basic CI/CD — Customer Churn Prediction

An end-to-end ML project demonstrating **Level 2 AI SDLC maturity**: modular code, automated CI/CD, model tracking with MLflow, containerized deployment, and automated testing — while data pipelines and evaluation remain manually triggered.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Operating Instructions](#operating-instructions)
- [Project Structure](#project-structure)
- [What Level 2 Fixes vs Level 1](#what-level-2-fixes-vs-level-1)
- [Compare with Level 1](#compare-with-level-1)
- [What Remains Manual](#what-remains-manual)

---

## Prerequisites

- **Python 3.10+** (3.12 recommended, pinned in `.python-version`)
- **UV** — fast Python package manager ([install guide](https://docs.astral.sh/uv/#installation))
- **Git** — for version control
- **Docker** *(optional)* — for containerized deployment

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Train the model
uv run python scripts/train.py

# 3. Run tests
uv run pytest tests/ -v

# 4. Start the prediction API
uv run python scripts/serve.py
```

---

## Operating Instructions

### 1. Environment Setup

All project dependencies are managed via `pyproject.toml`. UV handles virtual environment creation automatically:

```bash
uv sync --all-groups
```

This installs:
- **Runtime dependencies** (pandas, scikit-learn, flask, mlflow, etc.)
- **Dev dependencies** (pytest, ruff, mypy, jupyter)

### 2. Training Pipeline

```bash
uv run python scripts/train.py
```

**What it does:**
1. Loads `data/customer_data.csv` with schema validation
2. Builds a reusable `ColumnTransformer` preprocessing pipeline (scales numeric features; ordinal-encodes categoricals)
3. Splits data into train/test sets (80/20, seed from config)
4. Trains a `RandomForestClassifier` (hyperparams from config)
5. Logs all parameters, metrics, and the model artifact to MLflow (SQLite backend at `mlflow_data/mlflow.db`)
6. Saves `model.pkl` and `preprocessor.pkl` to `models/`

**Configuration** (all via environment variables or defaults):

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/customer_data.csv` | Path to training data |
| `MODEL_DIR` | `models/` | Output directory for serialized artifacts |
| `RANDOM_SEED` | `42` | Seed for train/test split and model |
| `TEST_SIZE` | `0.2` | Fraction of data reserved for testing |
| `N_ESTIMATORS` | `100` | Random forest tree count |
| `MAX_DEPTH` | `10` | Random forest max depth |
| `EXPERIMENT_NAME` | `churn-prediction` | MLflow experiment name |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Example with overrides:

```bash
$env:RANDOM_SEED = "7"; $env:N_ESTIMATORS = "200"; uv run python scripts/train.py
```

### 3. Evaluation Pipeline

```bash
uv run python scripts/evaluate.py
```

**What it does:**
1. Loads the saved `model.pkl` and `preprocessor.pkl`
2. Transforms the full dataset
3. Computes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
4. Saves all metrics to `models/metrics.json`

Example output:
```json
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "roc_auc": 1.0,
  "n_samples": 50,
  "confusion_matrix": [[30, 0], [0, 20]]
}
```

### 4. Model Serving API

Start the prediction server:

```bash
uv run python scripts/serve.py
```

Server starts on `http://localhost:8080` by default (set `PORT` to change).

#### Endpoints

**GET /health** — liveness probe

```bash
curl http://localhost:8080/health
```

Response:
```json
{"status": "ok", "model_loaded": true}
```

**POST /predict** — single or batch inference

Single customer:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "total_charges": 906.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "avg_monthly_usage_hours": 45,
    "late_payments_last_12m": 3
  }'
```

Batch prediction:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 35, "tenure_months": 12, "monthly_charges": 75.5, "total_charges": 906.0, "contract_type": "Month-to-month", "payment_method": "Electronic check", "internet_service": "Fiber optic", "tech_support": "No", "avg_monthly_usage_hours": 45, "late_payments_last_12m": 3},
    {"age": 56, "tenure_months": 48, "monthly_charges": 95.0, "total_charges": 4560.0, "contract_type": "Two year", "payment_method": "Credit card", "internet_service": "DSL", "tech_support": "Yes", "avg_monthly_usage_hours": 28, "late_payments_last_12m": 0}
  ]'
```

Response:
```json
{
  "predictions": [
    {"prediction": 1, "label": "CHURN", "confidence_stay": 0.15, "confidence_churn": 0.85},
    {"prediction": 0, "label": "STAY",  "confidence_stay": 0.92, "confidence_churn": 0.08}
  ]
}
```

### 5. Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src -v
```

**Test suite:**

| Test file | What it covers |
|---|---|
| `tests/test_features.py` | Preprocessor output shape, fitted vs transform consistency, unfitted error handling |
| `tests/test_evaluate.py` | Metric completeness, range validity, confusion matrix shape |

### 6. Viewing MLflow Runs

```bash
uv run mlflow ui --backend-store-uri ./mlflow_data
```

Open `http://localhost:5000` in a browser. Each training run logs:

- **Parameters**: `n_estimators`, `max_depth`, `random_seed`, `test_size`, `model_type`
- **Metrics**: `train_accuracy`, `test_accuracy`
- **Artifacts**: serialized model (skops format), conda environment, Python environment, pip requirements

### 7. Using the CI/CD Pipeline

The CI pipeline is defined in `.github/workflows/ci.yml`. It triggers on push/PR to `main`:

1. **lint-test** job: `ruff check` → `mypy` → `pytest`
2. **build** job (main only): Docker build and push to GitHub Container Registry

To use it:
1. Push code to GitHub
2. CI runs automatically
3. On `main`, a Docker image is published to `ghcr.io/<owner>/<repo>/churn-model:latest`

### 8. Docker Deployment

Build locally:

```bash
docker build -t churn-model:latest .
```

Run:

```bash
docker run -p 8080:8080 -v ./models:/app/models churn-model:latest
```

---

## Project Structure

```
Level_2_Basic_CI_CD/
├── README.md                       ← This file
├── ARCHITECTURE.md                 ← Architecture documentation
├── pyproject.toml                  ← Project metadata + dependencies
├── uv.lock                         ← Locked dependency tree (reproducible)
├── .python-version                 ← Python version pin for UV
├── .gitignore
├── .github/workflows/
│   └── ci.yml                      ← CI: lint → type-check → test → build
├── Dockerfile                      ← Multi-stage Docker build
├── src/                            ← Python package (importable)
│   ├── config.py                   ← Environment-variable config
│   ├── data/load.py                ← Data loading + schema validation
│   ├── features/build.py           ← ColumnTransformer preprocessing
│   ├── models/train.py             ← Model training + MLflow logging
│   └── evaluate.py                 ← Metric computation + JSON report
├── scripts/                        ← CLI entry points
│   ├── train.py                    ← uv run python scripts/train.py
│   ├── evaluate.py                 ← uv run python scripts/evaluate.py
│   └── serve.py                    ← WSGI entry for Flask app
├── app/
│   └── serve.py                    ← Flask app (predict + health endpoints)
├── tests/                          ← pytest suite
│   ├── conftest.py                 ← Shared fixtures (sample DataFrame)
│   ├── test_features.py
│   └── test_evaluate.py
├── notebooks/
│   └── explore.ipynb               ← EDA only — training moved to scripts
├── data/
│   └── customer_data.csv           ← 50 synthetic customer records
├── models/                         ← Generated artifacts (git-ignored)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
└── mlflow_data/                    ← MLflow SQLite store (git-ignored)
    └── mlflow.db
```

---

## What Level 2 Fixes vs Level 1

| Problem at Level 1 | How Level 2 Solves It | Where |
|---|---|---|
| **Notebook is the application** — training, eval, and visualization coupled in one file | Notebook is EDA-only; training and evaluation live in reusable Python scripts | `scripts/train.py`, `scripts/evaluate.py` |
| **Hardcoded paths** — every path breaks on move or clone | Config class reads from environment variables with sensible defaults | `src/config.py` |
| **No tests** — regressions reach production silently | pytest suite covering preprocessing pipeline and evaluation logic | `tests/` |
| **Fragile manual encoding** — unseen categories crash at runtime | `OrdinalEncoder` with `handle_unknown="use_encoded_value"` and `unknown_value=-1` | `src/features/build.py` |
| **Unlocked random seed** — train/test split changes each run | Seed in config, logged to MLflow — fully deterministic training | `src/config.py`, `src/models/train.py` |
| **No experiment tracking** — cannot tell which run produced a given model | MLflow logs parameters, metrics, and model artifact per run | `src/models/train.py` |
| **No CI pipeline** — code merges without automated validation | GitHub Actions: `ruff` lint → `mypy` type-check → `pytest` → Docker build | `.github/workflows/ci.yml` |
| **Manual SCP + SSH deployment** — fragile, no rollback, no audit | Docker image built and pushed to container registry; rollback via previous tag | `Dockerfile`, CI workflow |
| **No health endpoint** — can't tell if the service is alive | `/health` endpoint returns status and model-load confirmation | `app/serve.py` |
| **Model is a bare `.pkl` file** — no version, no lineage | MLflow run ID links model artifact to specific code, data, and hyperparameters | `src/models/train.py` |
| **No rollback strategy** — old `.pkl` may be lost | Container images are tagged; deploy any previous version instantly | CI pipeline tags |
| **No reproducibility** — cannot recreate a specific model | Locked seed + config snapshot + MLflow run ID = fully reproducible | `src/config.py` + MLflow |

---

## Compare with Level 1

| Aspect | Level 1 | Level 2 |
|---|---|---|
| **Code storage** | Notebooks shared via email/drive | Git version control with PRs |
| **Testing** | None | pytest (unit tests) |
| **CI/CD** | None | GitHub Actions (lint → type-check → test → build) |
| **Model tracking** | `.pkl` filename convention | MLflow (params, metrics, artifacts, run IDs) |
| **Preprocessing** | Manual `map()` in notebook cells | `ColumnTransformer` with `OrdinalEncoder` |
| **Serving** | Flask debug mode, no health check | Flask + `/health` + gunicorn-ready |
| **Deployment** | Manual SCP + SSH | Docker image + container registry |
| **Configuration** | Hardcoded strings | Environment variables with defaults |
| **Reproducibility** | Not possible | Locked seed + config + MLflow run ID |

---

## What Remains Manual

These are the **known limitations of Level 2** — they will be addressed in Level 3+.

- **Data preprocessing is manually triggered** — not part of CI/CD
- **Evaluation is not automated** — run manually after training
- **No monitoring or drift detection** — model degradation goes unnoticed until evaluated
- **Promotion to production requires manual approval** — no automated quality gates
- **Model performance screening is anecdotal** — relies on a single metrics snapshot
- **No automated retraining** — retraining must be manually triggered
