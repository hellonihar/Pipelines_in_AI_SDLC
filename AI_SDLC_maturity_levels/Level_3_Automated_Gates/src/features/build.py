import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline


CAT_ORDERINGS = {
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "internet_service": ["DSL", "Fiber optic"],
    "tech_support": ["No", "Yes"],
}


def get_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(
            categories=[CAT_ORDERINGS[col] for col in CAT_ORDERINGS],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, ["age", "tenure_months", "monthly_charges",
                                   "total_charges", "avg_monthly_usage_hours",
                                   "late_payments_last_12m"]),
        ("cat", categorical_pipeline, list(CAT_ORDERINGS.keys())),
    ])

    return preprocessor


def build_features(df: pd.DataFrame, preprocessor: ColumnTransformer, fit: bool = True):
    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"].values

    if fit:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    return X_transformed, y, preprocessor if fit else preprocessor
