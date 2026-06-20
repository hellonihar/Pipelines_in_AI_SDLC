# Business Requirements — Customer Churn Prediction

## 1. Business Problem

Customer churn costs the company an estimated $2.4M annually in lost recurring revenue. The organization currently has no systematic way to identify at-risk customers before they cancel. This project aims to build a churn prediction system that enables proactive retention interventions.

**Target:** Reduce voluntary churn by 15% within 6 months of deployment.

## 2. Stakeholders

| Role | Interest |
|------|----------|
| VP of Customer Success | Wants weekly list of at-risk accounts for outreach |
| Product Manager | Needs real-time scoring in the product UI |
| Finance / FP&A | Requires churn forecasts for quarterly planning |
| Data Science | Builds and maintains the model |
| Engineering | Deploys and serves the model in production |

## 3. Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | System shall predict churn probability (0.0–1.0) for each customer | P0 (Must) |
| FR-02 | System shall classify customers into risk tiers: Low (<0.3), Medium (0.3–0.7), High (>0.7) | P0 (Must) |
| FR-03 | System shall generate a batch churn report for all active customers weekly | P0 (Must) |
| FR-04 | System shall serve real-time predictions via a REST API | P1 (Should) |
| FR-05 | System shall return the top 3 factors contributing to a churn prediction | P2 (Nice) |
| FR-06 | System shall export at-risk customer list (High + Medium) as CSV | P1 (Should) |
| FR-07 | System shall log every prediction with customer ID, score, and timestamp | P1 (Should) |
| FR-08 | System shall support monthly retraining with updated customer data | P0 (Must) |

## 4. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Real-time API response time (p95) | < 500 ms |
| NFR-02 | Batch report generation (2,000 customers) | < 10 minutes |
| NFR-03 | Model accuracy (balanced) | ≥ 80% |
| NFR-04 | Model recall for churned class | ≥ 75% |
| NFR-05 | Service uptime (business hours) | ≥ 99.5% |
| NFR-06 | Predictions must be auditable — traceable to model version and input data | Required |
| NFR-07 | System must handle 5 concurrent API requests | Minimum |
| NFR-08 | Model training must complete | < 30 minutes |

## 5. Data Requirements

| Data Source | Description | Required? |
|-------------|-------------|-----------|
| Customer demographic data | Age, tenure, contract type, payment method | Yes |
| Usage data | Monthly charges, total charges, usage hours | Yes |
| Support interaction data | Number of support tickets, late payments | Yes |
| Churn labels | Historical churn flag (ground truth) | Yes |

**Volume:** ~2,000 active customers, ~50 feature columns.

## 6. Assumptions & Constraints

- **Assumption:** Historical churn labels are accurate and available for at least 12 months.
- **Assumption:** Customer demographic and usage data can be exported from the billing system weekly.
- **Constraint:** No real-time streaming infrastructure available — batch scoring is the initial delivery mechanism.
- **Constraint:** The model must be interpretable enough to explain predictions to non-technical stakeholders.
- **Assumption:** A churn definition ("no activity for 60+ days") is agreed upon by Product and CS teams.

## 7. Acceptance Criteria

1. A model exists that achieves ≥ 80% accuracy and ≥ 75% recall on held-out test data.
2. A weekly scheduled batch run produces a CSV of at-risk customers and delivers it to CS team.
3. A REST endpoint accepts customer features and returns churn probability in under 500ms (p95).
4. Predictions are logged with enough context to audit any single prediction.
5. The system can be retrained and redeployed in under 2 hours end-to-end.
6. A non-technical CS manager can interpret the output (risk tier + top factors).
