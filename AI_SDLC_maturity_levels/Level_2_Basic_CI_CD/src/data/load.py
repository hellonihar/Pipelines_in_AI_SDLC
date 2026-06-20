import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id", "age", "tenure_months", "monthly_charges",
    "total_charges", "contract_type", "payment_method",
    "internet_service", "tech_support", "avg_monthly_usage_hours",
    "late_payments_last_12m", "churned",
]

NUMERIC_COLUMNS = [
    "age", "tenure_months", "monthly_charges", "total_charges",
    "avg_monthly_usage_hours", "late_payments_last_12m",
]

CATEGORICAL_COLUMNS = [
    "contract_type", "payment_method", "internet_service", "tech_support",
]


def validate_schema(df: pd.DataFrame) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Schema valid — {len(REQUIRED_COLUMNS)} columns present")
    return df
