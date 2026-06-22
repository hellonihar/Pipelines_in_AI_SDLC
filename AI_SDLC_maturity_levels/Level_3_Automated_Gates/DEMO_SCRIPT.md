# Demo Script — Level 3: Automated Gates

**Audience:** Executives & Engineering Leadership  
**Duration:** 15–20 minutes  
**Goal:** Show how automated quality gates prevent bad data, bad models, and silent degradation from reaching production.

---

## Setup (do before the demo)

```bash
cd AI_SDLC_maturity_levels/Level_3_Automated_Gates
uv sync
uv run python scripts/train.py      # trains model + computes baseline stats
uv run python scripts/serve.py      # starts Flask app on http://localhost:8080
```

Pre-open `http://localhost:8080` in a browser tab. Pre-open the `monitoring/alerts.log` file in a text editor.

---

## 1. The Problem — Why Level 3 Matters (2 min)

| Pain Point (Level 2) | What Happens |
|---|---|
| No data validation | Bad data silently trains bad models |
| No evaluation gates | Regressions ship to production unnoticed |
| No production monitoring | Model degrades for weeks — users complain first |
| No drift detection | Distribution shifts cause silent failure |

**Say:**
> "At Level 2, we have CI/CD for code, but the model itself has no guardrails. Data quality is manual. Evaluation is manual. Monitoring doesn't exist. Level 3 adds automated gates at every stage — data in, model out, and ongoing production monitoring."

---

## 2. Dashboard — The Control Center (2 min)

**Navigate to:** `http://localhost:8080/`

**What to show:**
- Three gate cards (Data, Eval, Drift) — all showing `—` (not yet run)
- Latest Metrics panel — accuracy, precision, recall, F1, ROC AUC
- Telemetry Summary panel — prediction count, churn rate

**Say:**
> "This dashboard is the single pane of glass for model health. Three gates guard every stage of the ML lifecycle. Metrics and telemetry update live from the model and database. Let's run the gates."

**Action:** Click **"Run All Gates"**

**What happens:**
- Data Gate validates the training CSV (checks row count, nulls, anomalies, unseen categories)
- Eval Gate compares model metrics against configured thresholds
- Drift Gate checks if live prediction distributions have shifted from baseline
- All three gate cards flip to PASS/FAIL with detailed check tables

**Say:**
> "One click runs every gate. Each returns a structured pass/fail with per-check detail — not just a binary green/red, but exactly which check failed and by how much. No more digging through logs."

---

## 3. Configuration — Thresholds You Control (3 min)

**Navigate to:** `http://localhost:8080/config`

**Say:**
> "Gates are useless if you can't tune them. Every threshold lives in one place — editable from this UI."

**What to show:**
- **Data Validation:** `min_rows`, `max_null_fraction`, `max_anomaly_rate`, `max_novel_category_rate`
- **Evaluation:** `min_accuracy`, `min_precision`, `min_recall`, `min_f1`, `min_roc_auc`
- **Drift Detection:** `max_feature_drift_pvalue`, `max_psi`, `min_psi_samples`
- **Application Settings:** read-only view of data paths, ports, drift intervals

**Action:** Change `min_accuracy` from `0.75` to `0.95` and click **Save Changes**.

**Say:**
> "Suppose leadership decides model accuracy must be 95% to deploy. Our data science lead updates the threshold in 10 seconds and clicks Save. The next gate run enforces the new bar. No code change, no PR, no redeploy. But at 95%, our current model will fail — which means we either improve the model, or lower the threshold. That's a business decision, not a technical one."

**Action:** Change `min_accuracy` back to `0.75` and save.

---

## 4. Data Gate — Stop Bad Data Before Training (3 min)

**Navigate to:** `http://localhost:8080/gates/data`

**Say:**
> "The most expensive bug in ML is training on bad data. The Data Gate catches it before a single compute cycle is spent."

**What to show:**
- Run the gate using the sample data path
- The result shows pass/fail + detailed validation report

**Action:** Click **Run Data Gate**.

**Say:**
> "The gate checks four things: enough rows, low null rates, no excessive anomalies, and no unseen categories. The detail section tells you exactly which columns have issues."

**Optional — Show a deliberate failure:**
```bash
# In another terminal, create bad data
head -5 data/customer_data.csv > /tmp/bad_data.csv
echo "1001,34,12,75.5,906.0,NEW_CONTRACT_TYPE,,,No,45,3,1" >> /tmp/bad_data.csv
```
Then type `/tmp/bad_data.csv` into the CSV Path field and run.

**Say:**
> "Here, an upstream system sent a new contract type our model has never seen. The Data Gate flags it instantly — 'novel category rate exceeds threshold'. Training never starts. The team gets an alert instead of a wasted compute run."

---

## 5. Eval Gate — No Model Regressions (3 min)

**Navigate to:** `http://localhost:8080/gates/eval`

**Say:**
> "Once data passes, we train and evaluate. The Eval Gate ensures every new model meets our quality bar."

**Action:** Click **Run Eval Gate**.

**What to show:**
- Metric-by-metric comparison: value vs. threshold, with PASS/FAIL per metric
- Confusion matrix (TP, FP, FN, TN)
- Overall gate result

**Say:**
> "Every metric is compared against its threshold. If the model drops below on any metric — accuracy, recall, or ROC AUC — the gate fails and the model is rejected. CI won't deploy it. The team sees exactly what regressed."

**Point to the confusion matrix:**
> "This matters for stakeholder trust. We're not just looking at one number — we can see false positives vs. false negatives. In churn prediction, a false negative (missing a real churner) costs us a customer. The confusion matrix makes that visible."

---

## 6. Drift Gate — Detect Degradation in Production (3 min)

**Navigate to:** `http://localhost:8080/gates/drift`

**Say:**
> "The model passes data validation. It passes evaluation. It deploys to production. But customer behavior changes over time. The Drift Gate monitors live production data against the training baseline."

**What to show:**
- Telemetry summary (total predictions, churn rate, time range)
- Click **Check Drift Now** to trigger an on-demand drift check

**Say:**
> "The drift gate runs a background check every hour by default. You can also trigger it on demand. It uses two statistical methods — the Kolmogorov-Smirnov test for numeric features and Population Stability Index for categorical features."

**If the demo has telemetry data, show the per-feature drift table:**
- Each drifted feature shows p-value, KS statistic, live mean vs. baseline mean
- Overall PSI and drift summary

**Say:**
> "If the customer base shifts — say usage hours double or contract types change — this gate catches it. The alert system fires. The team knows within hours, not weeks. Without this, you discover degradation when a customer calls to complain."

---

## 7. Pipeline — Training End-to-End (2 min)

**Navigate to:** `http://localhost:8080/pipeline`

**Say:**
> "The full training pipeline integrates every gate. Let's run it end-to-end."

**Action:** Click **Run Full Pipeline**.

**What to show:**
- Live terminal output showing each step
  - Data validation → Data Gate
  - Feature engineering
  - Model training
  - Evaluation → Eval Gate
  - Baseline generation
- Pipeline passes only if all gates pass
- Model artifacts saved

**Say:**
> "Notice the Data Gate runs before training even starts. If data fails, we save the compute cost. The Eval Gate runs after evaluation — if metrics don't meet the bar, the model is rejected outright. This pipeline is what CI runs on every PR. Bad models never merge."

---

## 8. CI/CD Integration — Automated in GitHub (1 min)

Show the `.github/workflows/ci.yml` file or display the CI pipeline diagram from `ARCHITECTURE.md`.

**Say:**
> "This isn't just a local tool. The GitHub Actions workflow has three jobs: `lint-test` (code quality), `eval-gate` (train model + run gates), and `build` (Docker image). The build job only runs if the eval gate passes. A model regression blocks CI like a failing unit test would. That's the automation — no human needs to look at metrics and decide."

---

## 9. Architecture Diagram (1 min)

Display the architecture from `ARCHITECTURE.md` (the Mermaid diagram).

**Talking points:**
- **Data layer:** Raw data → Data Validator → Data Gate
- **Training layer:** Feature pipeline → Training → Evaluation → Eval Gate
- **CI/CD layer:** Only passes models that clear all gates
- **Serving layer:** Flask API + Telemetry Collector → SQLite DB
- **Monitoring layer:** Drift Detector (KS + PSI) → Alert Manager

**Say:**
> "Every arrow in this diagram represents an automated check. Data flows left to right. A failure at any gate stops the pipeline and writes to the alert log. This is the architecture that protects production models."

---

## 10. Business Value Recap (1 min)

| Capability | Before (Level 2) | After (Level 3) | Impact |
|---|---|---|---|
| Data quality | Manual inspection | Automated gate | Bad data caught before training |
| Model evaluation | Manual metrics check | Threshold gates in CI | Regressions block deployment |
| Production monitoring | None | Hourly drift + KS/PSI | Degradation found in hours |
| Alerting | None | File-based, extensible | Team knows when to act |
| Audit trail | MLflow run ID | Gate results + telemetry + alerts | Full provenance |

**Say:**
> "The business case estimates ~$261K/year in waste from data incidents, regression bugs, and undetected degradation. The Level 3 investment is ~$50K. That's a 4.5x ROI in Year 1 — and that's before counting the trust your team gains in production AI."

---

## Appendix: Key Files Referenced

| File | Purpose |
|---|---|
| `app/serve.py` | Flask server — API + UI routes + background drift check |
| `app/templates/*.html` | 7 HTML pages (dashboard, gates, config, pipeline) |
| `src/gates.py` | Gate logic — `run_data_gate`, `run_eval_gate`, `run_drift_gate` |
| `config/gates.yaml` | Configurable gate thresholds |
| `src/data/validate.py` | Statistical data validation (nulls, anomalies, novel cats) |
| `src/monitoring/drift.py` | KS test + PSI drift detection |
| `src/monitoring/alert.py` | File-based alerting (INFO/WARNING/CRITICAL) |
| `src/monitoring/collector.py` | Prediction telemetry → SQLite |
| `scripts/train.py` | Training pipeline with gates |
| `scripts/serve.py` | App entry point |
| `.github/workflows/ci.yml` | CI with eval gate job |
