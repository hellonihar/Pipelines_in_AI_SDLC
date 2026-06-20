import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": [1001, 1002, 1003],
        "age": [34, 56, 28],
        "tenure_months": [12, 48, 6],
        "monthly_charges": [75.5, 95.0, 45.0],
        "total_charges": [906.0, 4560.0, 270.0],
        "contract_type": ["Month-to-month", "Two year", "Month-to-month"],
        "payment_method": ["Electronic check", "Credit card", "Mailed check"],
        "internet_service": ["Fiber optic", "DSL", "DSL"],
        "tech_support": ["No", "Yes", "No"],
        "avg_monthly_usage_hours": [45, 28, 55],
        "late_payments_last_12m": [3, 0, 5],
        "churned": [1, 0, 1],
    })


@pytest.fixture
def drift_report():
    return {
        "drifted_features": {
            "age": {"type": "numeric", "p_value": 0.002, "ks_stat": 0.45},
        },
        "overall_psi": 0.35,
    }
