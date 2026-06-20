from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.load import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


@dataclass
class DataValidationReport:
    passed: bool = False
    n_rows: int = 0
    n_columns: int = 0
    null_fractions: dict[str, float] = field(default_factory=dict)
    anomaly_rates: dict[str, float] = field(default_factory=dict)
    novel_categories: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def check_nulls(df: pd.DataFrame) -> dict[str, float]:
    return df.isnull().mean().to_dict()


def check_anomalies(df: pd.DataFrame, num_cols: list[str] | None = None) -> dict[str, float]:
    if num_cols is None:
        num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    rates = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            rates[col] = 0.0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outside = ((df[col] < lower) | (df[col] > upper)).sum()
        rates[col] = round(n_outside / len(df), 4)
    return rates


def check_novel_categories(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    known_categories: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    novel = {}
    for col in cat_cols:
        vals = df[col].dropna().unique().tolist()
        if known_categories and col in known_categories:
            unknowns = [v for v in vals if v not in known_categories[col]]
            if unknowns:
                novel[col] = unknowns
    return novel


def validate_data(
    df: pd.DataFrame,
    min_rows: int = 10,
    max_null_fraction: float = 0.05,
    max_anomaly_rate: float = 0.10,
    max_novel_category_rate: float = 0.20,
    known_categories: dict[str, list[str]] | None = None,
) -> DataValidationReport:
    report = DataValidationReport(
        n_rows=len(df),
        n_columns=len(df.columns),
    )

    missing = [c for c in df.columns if df[c].isnull().all()]
    if missing:
        report.errors.append(f"Columns entirely null: {missing}")

    null_fracs = check_nulls(df)
    report.null_fractions = null_fracs
    for col, frac in null_fracs.items():
        if frac > max_null_fraction:
            report.warnings.append(
                f"Column '{col}' has {frac:.1%} nulls (threshold: {max_null_fraction:.0%})"
            )

    anomaly_rates = check_anomalies(df)
    report.anomaly_rates = anomaly_rates
    for col, rate in anomaly_rates.items():
        if rate > max_anomaly_rate:
            report.warnings.append(
                f"Column '{col}' has {rate:.1%} anomalies (threshold: {max_anomaly_rate:.0%})"
            )

    novel = check_novel_categories(df, known_categories=known_categories)
    report.novel_categories = novel
    total_novel = sum(len(v) for v in novel.values())
    novel_rate = total_novel / len(CATEGORICAL_COLUMNS) if CATEGORICAL_COLUMNS else 0
    if novel_rate > max_novel_category_rate:
        report.warnings.append(
            f"{total_novel} unseen categories found across features (rate: {novel_rate:.0%})"
        )

    for col in report.warnings:
        print(f"  WARNING: {col}")

    report.passed = len(report.errors) == 0
    return report
