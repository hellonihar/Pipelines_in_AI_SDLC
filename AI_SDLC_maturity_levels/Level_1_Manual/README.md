# Level 1: Manual — Example Application

## Customer Churn Prediction

This is a deliberately **Level 1** example application that demonstrates how AI delivery works without pipelines, automation, or reproducibility controls. It illustrates the pain points quantified in the business case.

---

## What's Here

```
Level_1_Manual/
├── README.md               ← This file
├── business_case.md         ← Business case for Level 1 → Level 2
├── requirements.txt         ← Python dependencies
├── data/
│   └── customer_data.csv    ← 50 synthetic customer records
├── notebooks/
│   └── churn_model_dev.ipynb ← Jupyter notebook with full workflow
├── models/
│   └── (generated .pkl files land here)
├── scripts/
│   └── deploy.ps1           ← Manual deployment script (SCP + SSH)
└── app/
    └── serve_model.py       ← Minimal Flask model server
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Open `notebooks/churn_model_dev.ipynb` in Jupyter and run all cells. This will:

- Load `data/customer_data.csv`
- Preprocess features with hardcoded mappings
- Train a random forest classifier
- Print evaluation metrics
- Export a `.pkl` file to `models/`

### 3. Deploy manually (simulated)

```powershell
.\scripts\deploy.ps1 -ModelPath ".\models\churn_model_2026-06-20.pkl" -Server "prod-serve-01"
```

*Requires a target server with SSH access.*

### 4. Serve locally

```bash
python app/serve_model.py
```

Test with:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"tenure_months":12,"monthly_charges":75.5,"total_charges":906,"contract_type":0,"payment_method":0,"internet_service":1,"tech_support":0,"avg_monthly_usage_hours":45,"late_payments_last_12m":3}'
```

---

## What This Demonstrates

### Level 1 Characteristics

| File | Characteristic | Why It's Fragile |
|------|---------------|------------------|
| `churn_model_dev.ipynb` | Everything in one notebook | Can't reuse preprocessing; can't schedule retraining |
| `churn_model_dev.ipynb` | Hardcoded data path | Breaks if `customer_data.csv` moves or is renamed |
| `churn_model_dev.ipynb` | Manual encoding | Unseen categories (e.g. a new contract type) crash training |
| `churn_model_dev.ipynb` | Random seed not locked | Different split every run → metrics not reproducible |
| `churn_model_dev.ipynb` | Hyperparameters guessed | No record of why n_estimators=100 or max_depth=10 |
| `churn_model_dev.ipynb` | No evaluation saved | Run the notebook, see metrics, close it — metrics are gone |
| `deploy.ps1` | Manual SCP + SSH | No rollback, no canary, no health check before cutover |
| `deploy.ps1` | Hardcoded server name | If the server changes, the script must be edited |
| Models exported as `.pkl` | No versioning | Two models with the same name can coexist — which is live? |
| `serve_model.py` | Debug mode in production | Crashes silently, no monitoring, no graceful shutdown |

### What's Missing (Compared to Level 2+)

- No CI/CD pipeline
- No automated tests
- No model registry
- No experiment tracking
- No data validation gates
- No monitoring or drift detection
- No reproducible training
- No audit trail

---

## Intended Use

This example is a **teaching tool** — it exists to be compared with the later maturity levels so you can see the concrete difference that pipelines, automation, and reproducibility make.

Refer to the business case in this folder for the cost-benefit analysis of moving from Level 1 to Level 2.
