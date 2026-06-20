# Level 3 — Automated Gates

AI SDLC maturity level with automated quality gates, monitoring, and drift detection running against the same customer churn dataset as levels 1–2.

## Directory Layout

```
Level_3_Automated_Gates/
├── config/            # Gate thresholds (gates.yaml)
├── data/              # Training dataset
├── monitoring/        # Telemetry DB & baseline stats
├── notebooks/         # Exploratory analysis
├── scripts/           # train, evaluate, serve, validate, gates, drift
├── src/               # Reusable package (data, features, models, monitoring, gates)
├── tests/             # Pytest suite
├── pyproject.toml
├── Dockerfile
├── .github/workflows/ # CI with gate checks
├── ARCHITECTURE.md
└── BUSINESS_CASE.md
```

## Quick Start

```bash
uv sync
uv run python scripts/train.py          # Train with gates
uv run python scripts/check_drift.py     # Check prediction drift
uv run python scripts/serve.py           # Start prediction API
uv run pytest tests/ -v                  # Run tests
```

## What's New vs Level 2

- Automated quality gates (data validation, evaluation thresholds, drift detection)
- Monitoring with SQLite telemetry and periodic drift checks
- Alert system (file-based, extensible to email/Slack)
- Configurable thresholds in `config/gates.yaml`
- Background drift checker in the serving app
- The serving app includes `/monitoring/summary` and `/monitoring/drift` endpoints

## Gate Configuration

Edit `config/gates.yaml` to tune thresholds:

```yaml
data_validation:
  min_rows: 50
  max_null_fraction: 0.05
  max_anomaly_rate: 0.05

evaluation:
  min_accuracy: 0.75
  min_roc_auc: 0.75

drift:
  max_feature_drift_pvalue: 0.05
  max_psi: 0.20
```
