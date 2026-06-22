from dataclasses import dataclass, field

import yaml


@dataclass
class GateResult:
    passed: bool
    gate_name: str
    checks: dict = field(default_factory=dict)
    summary: str = ""


def load_gate_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_data_gate(thresholds: dict, report) -> GateResult:
    checks = {}
    warnings = []

    if report.n_rows < thresholds.get("min_rows", 10):
        checks["min_rows"] = False
        warnings.append(f"Only {report.n_rows} rows (min: {thresholds['min_rows']})")
    else:
        checks["min_rows"] = True

    high_null_cols = [
        (c, v) for c, v in report.null_fractions.items()
        if v > thresholds.get("max_null_fraction", 0.05)
    ]
    if high_null_cols:
        checks["null_fraction"] = False
        for col, val in high_null_cols:
            warnings.append(f"Column '{col}' null fraction {val:.2%}")
    else:
        checks["null_fraction"] = True

    high_anom_cols = [
        (c, v) for c, v in report.anomaly_rates.items()
        if v > thresholds.get("max_anomaly_rate", 0.10)
    ]
    if high_anom_cols:
        checks["anomaly_rate"] = False
        for col, val in high_anom_cols:
            warnings.append(f"Column '{col}' anomaly rate {val:.2%}")
    else:
        checks["anomaly_rate"] = True

    total_novel = sum(len(v) for v in report.novel_categories.values())
    feature_count = max(len(report.novel_categories), 1)
    novel_rate = total_novel / feature_count
    max_novel = thresholds.get("max_novel_category_rate", 0.20)
    if novel_rate > max_novel:
        checks["novel_categories"] = False
        warnings.append(f"Novel category rate {novel_rate:.2%} (max: {max_novel:.0%})")
    else:
        checks["novel_categories"] = True

    passed = all(checks.values())
    summary = "; ".join(warnings) if warnings else "All data checks passed"
    return GateResult(passed=passed, gate_name="data_validation", checks=checks, summary=summary)


def run_eval_gate(thresholds: dict, metrics: dict) -> GateResult:
    checks = {}
    warnings = []

    for key, threshold in [
        ("accuracy", "min_accuracy"),
        ("precision", "min_precision"),
        ("recall", "min_recall"),
        ("f1_score", "min_f1"),
        ("roc_auc", "min_roc_auc"),
    ]:
        if key not in metrics:
            checks[key] = False
            warnings.append(f"Metric '{key}' missing")
            continue
        min_val = thresholds.get(threshold, 0.0)
        if metrics[key] < min_val:
            checks[key] = False
            warnings.append(f"{key}={metrics[key]:.4f} < {min_val}")
        else:
            checks[key] = True

    passed = all(checks.values())
    summary = "; ".join(warnings) if warnings else "All evaluation checks passed"
    return GateResult(passed=passed, gate_name="evaluation", checks=checks, summary=summary)


def run_all_gates(gate_cfg: dict, data_report=None, metrics=None, drift_report=None) -> list[GateResult]:
    results = []
    if data_report is not None:
        results.append(run_data_gate(gate_cfg.get("data_validation", {}), data_report))
    if metrics is not None:
        results.append(run_eval_gate(gate_cfg.get("evaluation", {}), metrics))
    if drift_report is not None:
        results.append(run_drift_gate(gate_cfg.get("drift", {}), drift_report))
    return results


def run_drift_gate(thresholds: dict, drift_report: dict) -> GateResult:
    checks = {}
    warnings = []

    drifted = drift_report.get("drifted_features", {})
    for feat, details in drifted.items():
        pvalue = details.get("p_value", 1.0)
        max_p = thresholds.get("max_feature_drift_pvalue", 0.05)
        if pvalue < max_p:
            checks[f"drift_{feat}"] = False
            warnings.append(f"Feature '{feat}' drifted (p={pvalue:.4f})")

    psi = drift_report.get("overall_psi", 0.0)
    max_psi = thresholds.get("max_psi", 0.20)
    if psi > max_psi:
        checks["overall_psi"] = False
        warnings.append(f"Overall PSI {psi:.4f} exceeds {max_psi}")

    if not checks:
        checks["all_features"] = True

    passed = all(checks.values()) if checks else True
    summary = "; ".join(warnings) if warnings else "No drift detected"
    return GateResult(passed=passed, gate_name="drift", checks=checks, summary=summary)
