import pandas as pd
import numpy as np
from src.data.validate import (
    validate_data, check_nulls, check_anomalies, check_novel_categories,
)
from src.features.build import CAT_ORDERINGS


class TestValidate:
    def test_passes_clean_data(self, sample_df):
        report = validate_data(sample_df)
        assert report.passed
        assert report.n_rows == 3
        assert len(report.warnings) == 0

    def test_detects_null_fraction(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "age"] = np.nan
        df.loc[1, "age"] = np.nan
        report = validate_data(df, max_null_fraction=0.01)
        assert any("null" in w for w in report.warnings)

    def test_detects_anomalies(self):
        df = pd.DataFrame({
            "age": [30, 35, 32, 31, 1000],
            "tenure_months": [12, 15, 14, 13, 20],
            "monthly_charges": [50, 55, 52, 53, 60],
            "total_charges": [600, 800, 700, 650, 1000],
            "avg_monthly_usage_hours": [40, 42, 41, 39, 45],
            "late_payments_last_12m": [0, 1, 0, 0, 1],
            "contract_type": ["A", "B", "A", "B", "A"],
            "payment_method": ["X", "Y", "X", "Y", "X"],
            "internet_service": ["M", "N", "M", "N", "M"],
            "tech_support": ["Yes", "No", "Yes", "No", "Yes"],
            "customer_id": [1, 2, 3, 4, 5],
            "churned": [0, 1, 0, 1, 0],
        })
        report = validate_data(df, max_anomaly_rate=0.01)
        assert report.anomaly_rates.get("age", 0) > 0.01

    def test_check_nulls(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "age"] = np.nan
        nulls = check_nulls(df)
        assert nulls["age"] > 0

    def test_check_novel_categories(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "contract_type"] = "NEW_TYPE"
        novel = check_novel_categories(df, known_categories=CAT_ORDERINGS)
        assert "contract_type" in novel
        assert "NEW_TYPE" in novel["contract_type"]
