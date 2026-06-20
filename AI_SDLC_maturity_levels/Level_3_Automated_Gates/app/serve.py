import os
import sys
import threading
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.config import Config
from src.features.build import CAT_ORDERINGS
from src.monitoring.collector import TelemetryCollector
from src.monitoring.drift import check_drift
from src.monitoring.alert import AlertManager
from src.gates import load_gate_config, run_drift_gate

config = Config()
app = Flask(__name__)

model_path = os.getenv("MODEL_PATH", f"{config.model_dir}/model.pkl")
preprocessor_path = os.getenv("PREPROCESSOR_PATH", f"{config.model_dir}/preprocessor.pkl")

if not os.path.exists(model_path):
    print(f"FATAL: Model not found at {model_path}", file=sys.stderr)
    sys.exit(1)

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

collector = TelemetryCollector(config.telemetry_db, config.monitoring_retention_days)
alert = AlertManager(config.alerts_log)
gate_cfg = load_gate_config(config.gates_config)

feature_names = (
    ["age", "tenure_months", "monthly_charges", "total_charges",
     "avg_monthly_usage_hours", "late_payments_last_12m"]
    + list(CAT_ORDERINGS.keys())
)

_cached_drift = {"report": None, "timestamp": None}

print(f"Model loaded from {model_path}")
print(f"Telemetry DB: {config.telemetry_db}")
print(f"Alerts log: {config.alerts_log}")
print(f"Expected features: {feature_names}")


def _drift_background_check():
    while True:
        time.sleep(config.drift_check_interval)
        try:
            live_df = collector.get_predictions_since(config.drift_window_hours)
            if len(live_df) < 10:
                continue

            report = check_drift(
                config.baseline_stats_path, live_df,
                p_threshold=gate_cfg.get("drift", {}).get("max_feature_drift_pvalue", 0.05),
                psi_threshold=gate_cfg.get("drift", {}).get("max_psi", 0.20),
                min_samples=gate_cfg.get("drift", {}).get("min_psi_samples", 30),
            )

            _cached_drift["report"] = {
                "passed": report.passed,
                "drifted_features": report.drifted_features,
                "overall_psi": report.overall_psi,
                "n_drifted": report.n_drifted,
                "n_features": report.n_features,
                "timestamp": report.timestamp,
            }
            _cached_drift["timestamp"] = report.timestamp

            gate = run_drift_gate(gate_cfg.get("drift", {}), {
                "drifted_features": report.drifted_features,
                "overall_psi": report.overall_psi,
            })
            alert.alert_from_drift(report)

        except Exception as e:
            print(f"Drift check error: {e}")


threading.Thread(target=_drift_background_check, daemon=True).start()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Expected object or array"}), 400

    try:
        X = df[feature_names]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    X_processed = preprocessor.transform(X)
    preds = model.predict(X_processed).tolist()
    probs = model.predict_proba(X_processed).tolist()

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "prediction": int(pred),
            "label": "CHURN" if pred == 1 else "STAY",
            "confidence_stay": round(prob[0], 4),
            "confidence_churn": round(prob[1], 4),
        })
        collector.log_prediction(
            {k: df.iloc[i][k] for k in feature_names},
            int(pred), round(prob[1], 4), round(prob[0], 4),
        )

    return jsonify({"predictions": results})


@app.route("/monitoring/summary", methods=["GET"])
def monitoring_summary():
    return jsonify(collector.get_summary())


@app.route("/monitoring/drift", methods=["GET"])
def monitoring_drift():
    cached = _cached_drift["report"]
    if cached is None:
        return jsonify({"status": "pending", "message": "Drift check not yet run"})
    return jsonify(cached)


if __name__ == "__main__":
    port = config.port
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=(config.log_level == "DEBUG"))
