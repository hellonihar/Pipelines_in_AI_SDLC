import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class DriftReport:
    passed: bool = False
    drifted_features: dict = field(default_factory=dict)
    overall_psi: float = 0.0
    n_drifted: int = 0
    n_features: int = 0
    timestamp: str = ""


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    expected_hist, edges = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=edges)

    n_expected = expected_hist.sum()
    n_actual = actual_hist.sum()

    if n_expected == 0 or n_actual == 0:
        return float("inf")

    expected_pct = expected_hist / n_expected
    actual_pct = actual_hist / n_actual

    expected_pct = np.clip(expected_pct, eps, 1 - eps)
    actual_pct = np.clip(actual_pct, eps, 1 - eps)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_categorical_psi(expected_counts: dict, actual_counts: dict) -> float:
    eps = 1e-6
    all_keys = set(expected_counts.keys()) | set(actual_counts.keys())
    expected_total = sum(expected_counts.values())
    actual_total = sum(actual_counts.values())

    psi = 0.0
    for key in all_keys:
        p = (expected_counts.get(key, 0) + eps) / (expected_total + eps * len(all_keys))
        q = (actual_counts.get(key, 0) + eps) / (actual_total + eps * len(all_keys))
        psi += (q - p) * np.log(q / p)
    return psi


def check_drift(
    baseline_path: str,
    live_df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    p_threshold: float = 0.05,
    psi_threshold: float = 0.20,
    min_samples: int = 30,
) -> DriftReport:
    with open(baseline_path) as f:
        baseline = json.load(f)

    if numeric_features is None:
        numeric_features = list(baseline.get("numeric", {}).keys())
    if categorical_features is None:
        categorical_features = list(baseline.get("categorical", {}).keys())

    report = DriftReport(
        n_features=len(numeric_features) + len(categorical_features),
        timestamp=datetime.utcnow().isoformat(),
    )

    if len(live_df) < min_samples:
        report.passed = True
        report.drifted_features = {}
        return report

    drifted = {}

    for feat in numeric_features:
        if feat not in live_df.columns:
            continue
        baseline_stats = baseline.get("numeric", {}).get(feat)
        if not baseline_stats:
            continue

        live_values = live_df[feat].dropna().values
        baseline_values_for_test = np.random.default_rng(42).normal(
            baseline_stats["mean"], baseline_stats["std"],
            size=min(len(live_values), 1000),
        )

        stat, p_value = ks_2samp(baseline_values_for_test, live_values)
        if p_value < p_threshold:
            drifted[feat] = {
                "type": "numeric",
                "p_value": round(p_value, 4),
                "ks_stat": round(stat, 4),
                "live_mean": round(float(live_values.mean()), 2),
                "baseline_mean": baseline_stats["mean"],
            }

    cat_baseline = baseline.get("categorical", {})
    for feat in categorical_features:
        if feat not in live_df.columns or feat not in cat_baseline:
            continue
        live_counts = live_df[feat].value_counts().to_dict()
        psi = compute_categorical_psi(cat_baseline[feat], live_counts)
        if psi > psi_threshold:
            drifted[feat] = {
                "type": "categorical",
                "psi": round(psi, 4),
            }

    report.drifted_features = drifted
    report.n_drifted = len(drifted)
    report.overall_psi = round(
        compute_psi_for_report(baseline, live_df, numeric_features), 4
    )
    report.passed = report.n_drifted == 0

    return report


def compute_psi_for_report(baseline: dict, live_df: pd.DataFrame, numeric_features: list[str]) -> float:
    psi_vals = []
    for feat in numeric_features:
        if feat not in live_df.columns:
            continue
        base_stats = baseline.get("numeric", {}).get(feat)
        if not base_stats:
            continue
        live_vals = live_df[feat].dropna().values
        if len(live_vals) < 10:
            continue
        expected = np.random.default_rng(42).normal(
            base_stats["mean"], base_stats["std"], size=len(live_vals),
        )
        psi_vals.append(compute_psi(expected, live_vals))
    return float(np.mean(psi_vals)) if psi_vals else 0.0
