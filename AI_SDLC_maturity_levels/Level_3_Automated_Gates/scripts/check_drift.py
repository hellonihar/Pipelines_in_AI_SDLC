import argparse

from src.config import Config
from src.gates import load_gate_config, run_drift_gate
from src.monitoring.collector import TelemetryCollector
from src.monitoring.drift import check_drift
from src.monitoring.alert import AlertManager


def main():
    parser = argparse.ArgumentParser(description="Check for data drift in prediction telemetry")
    parser.add_argument("--config", help="Path to gates.yaml")
    parser.add_argument("--hours", type=int, help="Hours of telemetry to examine")
    args = parser.parse_args()

    config = Config()

    gate_path = args.config or config.gates_config
    gate_cfg = load_gate_config(gate_path)
    drift_cfg = gate_cfg.get("drift", {})

    hours = args.hours or config.drift_window_hours

    collector = TelemetryCollector(config.telemetry_db)
    live_df = collector.get_predictions_since(hours)

    print(f"=== Drift Check ===")
    print(f"Baseline: {config.baseline_stats_path}")
    print(f"Live data: {len(live_df)} predictions from last {hours}h")
    print()

    if len(live_df) < drift_cfg.get("min_psi_samples", 30):
        print(f"Not enough data for drift detection ({len(live_df)} < {drift_cfg.get('min_psi_samples', 30)})")
        return 0

    report = check_drift(
        config.baseline_stats_path, live_df,
        p_threshold=drift_cfg.get("max_feature_drift_pvalue", 0.05),
        psi_threshold=drift_cfg.get("max_psi", 0.20),
        min_samples=drift_cfg.get("min_psi_samples", 30),
    )

    gate = run_drift_gate(drift_cfg, {
        "drifted_features": report.drifted_features,
        "overall_psi": report.overall_psi,
    })

    alert = AlertManager(config.alerts_log)
    alert.alert_from_drift(report)

    print(f"Drifted features: {report.n_drifted}/{report.n_features}")
    print(f"Overall PSI: {report.overall_psi:.4f}")
    print(f"Gate: {'PASSED' if gate.passed else 'FAILED'} — {gate.summary}")

    return 0 if gate.passed else 1


if __name__ == "__main__":
    exit(main())
