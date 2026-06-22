import io
import json
import os
import sys
import threading
import time
import yaml

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from pathlib import Path

from src.config import Config
from src.data.load import load_data, CATEGORICAL_COLUMNS
from src.data.validate import validate_data, DataValidationReport
from src.evaluate import evaluate_model
from src.features.build import build_features, get_preprocessor, CAT_ORDERINGS
from src.models.train import train_model
from src.monitoring.collector import TelemetryCollector
from src.monitoring.drift import check_drift
from src.monitoring.alert import AlertManager
from src.gates import load_gate_config, run_data_gate, run_eval_gate, run_drift_gate, run_all_gates

config = Config()
app = Flask(__name__)

model_path = os.getenv("MODEL_PATH", f"{config.model_dir}/model.pkl")
preprocessor_path = os.getenv("PREPROCESSOR_PATH", f"{config.model_dir}/preprocessor.pkl")

model_loaded = False
model = None
preprocessor = None

if os.path.exists(model_path) and os.path.exists(preprocessor_path):
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        model_loaded = True
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}", file=sys.stderr)
else:
    print(f"WARNING: Model not found at {model_path} — serving will be limited", file=sys.stderr)

collector = TelemetryCollector(config.telemetry_db, config.monitoring_retention_days)
alert = AlertManager(config.alerts_log)
gate_cfg = load_gate_config(config.gates_config)

feature_names = (
    ["age", "tenure_months", "monthly_charges", "total_charges",
     "avg_monthly_usage_hours", "late_payments_last_12m"]
    + list(CAT_ORDERINGS.keys())
)

_cached_drift = {"report": None, "timestamp": None}

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


# ---------------------------------------------------------------------------
# Existing prediction & monitoring endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded})


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

    if not model_loaded or model is None or preprocessor is None:
        return jsonify({"error": "Model not loaded"}), 503

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


# ---------------------------------------------------------------------------
# UI routes — HTML pages
# ---------------------------------------------------------------------------

@app.route("/")
def ui_dashboard():
    return render_template("dashboard.html", data_path=config.data_path)


@app.route("/gates/data")
def ui_data_gate():
    return render_template("data_gate.html", data_path=config.data_path)


@app.route("/gates/eval")
def ui_eval_gate():
    return render_template("eval_gate.html")


@app.route("/gates/drift")
def ui_drift_gate():
    return render_template("drift_gate.html")


@app.route("/pipeline")
def ui_pipeline():
    return render_template("pipeline.html")


@app.route("/config")
def ui_config():
    return render_template("config.html")


# ---------------------------------------------------------------------------
# API: Evaluation metrics
# ---------------------------------------------------------------------------

@app.route("/api/evaluation/metrics", methods=["GET"])
def api_evaluation_metrics():
    metrics_path = os.path.join(config.model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "No metrics found"}), 404
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["thresholds"] = gate_cfg.get("evaluation", {})
    return jsonify(metrics)


# ---------------------------------------------------------------------------
# API: Gate execution
# ---------------------------------------------------------------------------

@app.route("/api/gates/data", methods=["POST"])
def api_gate_data():
    body = request.get_json() or {}
    data_path = body.get("path", config.data_path)

    if not os.path.exists(data_path):
        return jsonify({"error": f"File not found: {data_path}"}), 400

    try:
        df = load_data(data_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {e}"}), 400

    known_categories = {col: list(vals) for col, vals in CAT_ORDERINGS.items()}

    report = validate_data(
        df,
        known_categories=known_categories,
        **gate_cfg.get("data_validation", {}),
    )

    gate = run_data_gate(gate_cfg.get("data_validation", {}), report)

    return jsonify({
        "passed": gate.passed,
        "gate_name": gate.gate_name,
        "checks": gate.checks,
        "summary": gate.summary,
        "report": {
            "passed": report.passed,
            "n_rows": report.n_rows,
            "n_columns": report.n_columns,
            "warnings": report.warnings,
            "errors": report.errors,
            "null_fractions": report.null_fractions,
            "anomaly_rates": report.anomaly_rates,
            "novel_categories": report.novel_categories,
        },
    })


@app.route("/api/gates/eval", methods=["POST"])
def api_gate_eval():
    metrics_path = os.path.join(config.model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "No evaluation metrics found. Train a model first."}), 404

    with open(metrics_path) as f:
        metrics = json.load(f)

    gate = run_eval_gate(gate_cfg.get("evaluation", {}), metrics)

    thresholds = gate_cfg.get("evaluation", {})
    per_metric = {}
    for key, th_key in [
        ("accuracy", "min_accuracy"),
        ("precision", "min_precision"),
        ("recall", "min_recall"),
        ("f1_score", "min_f1"),
        ("roc_auc", "min_roc_auc"),
    ]:
        if key in metrics:
            per_metric[key] = {
                "value": metrics[key],
                "threshold": thresholds.get(th_key),
                "passed": metrics[key] >= thresholds.get(th_key, 0) if th_key in thresholds else None,
            }

    return jsonify({
        "passed": gate.passed,
        "gate_name": gate.gate_name,
        "checks": gate.checks,
        "summary": gate.summary,
        "metrics": per_metric,
        "thresholds": thresholds,
    })


@app.route("/api/gates/drift", methods=["POST"])
def api_gate_drift():
    body = request.get_json() or {}
    hours = body.get("hours", config.drift_window_hours)

    if not os.path.exists(config.baseline_stats_path):
        return jsonify({"error": "No baseline stats found. Train a model first."}), 404

    live_df = collector.get_predictions_since(hours)
    drift_thresholds = gate_cfg.get("drift", {})

    if len(live_df) < drift_thresholds.get("min_psi_samples", 30):
        return jsonify({
            "passed": True,
            "gate_name": "drift",
            "checks": {"all_features": True},
            "summary": f"Insufficient data ({len(live_df)} samples, need {drift_thresholds.get('min_psi_samples', 30)})",
            "drift_report": None,
        })

    report = check_drift(
        config.baseline_stats_path, live_df,
        p_threshold=drift_thresholds.get("max_feature_drift_pvalue", 0.05),
        psi_threshold=drift_thresholds.get("max_psi", 0.20),
        min_samples=drift_thresholds.get("min_psi_samples", 30),
    )

    gate = run_drift_gate(drift_thresholds, {
        "drifted_features": report.drifted_features,
        "overall_psi": report.overall_psi,
    })

    _cached_drift["report"] = {
        "passed": report.passed,
        "drifted_features": report.drifted_features,
        "overall_psi": report.overall_psi,
        "n_drifted": report.n_drifted,
        "n_features": report.n_features,
        "timestamp": report.timestamp,
    }
    _cached_drift["timestamp"] = report.timestamp

    return jsonify({
        "passed": gate.passed,
        "gate_name": gate.gate_name,
        "checks": gate.checks,
        "summary": gate.summary,
        "drift_report": {
            "passed": report.passed,
            "drifted_features": report.drifted_features,
            "overall_psi": report.overall_psi,
            "n_drifted": report.n_drifted,
            "n_features": report.n_features,
            "timestamp": report.timestamp,
        },
    })


@app.route("/api/gates/all", methods=["POST"])
def api_gate_all():
    body = request.get_json() or {}
    data_path = body.get("data_path", config.data_path)
    results = []
    logs = []

    # Data gate
    if os.path.exists(data_path):
        try:
            df = load_data(data_path)
            known_categories = {col: list(vals) for col, vals in CAT_ORDERINGS.items()}
            report = validate_data(df, known_categories=known_categories, **gate_cfg.get("data_validation", {}))
            dg = run_data_gate(gate_cfg.get("data_validation", {}), report)
            data_gate = {"passed": dg.passed, "gate_name": dg.gate_name, "checks": dg.checks, "summary": dg.summary}
            logs.append(f"Data Gate: {'PASS' if dg.passed else 'FAIL'} — {dg.summary}")
        except Exception as e:
            data_gate = {"passed": False, "gate_name": "data_validation", "checks": {}, "summary": f"Error: {e}"}
            logs.append(f"Data Gate: ERROR — {e}")
    else:
        data_gate = {"passed": False, "gate_name": "data_validation", "checks": {}, "summary": f"File not found: {data_path}"}
        logs.append(f"Data Gate: SKIP — file not found")

    # Eval gate
    metrics_path = os.path.join(config.model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        eg = run_eval_gate(gate_cfg.get("evaluation", {}), metrics)
        eval_gate = {"passed": eg.passed, "gate_name": eg.gate_name, "checks": eg.checks, "summary": eg.summary}
        logs.append(f"Eval Gate: {'PASS' if eg.passed else 'FAIL'} — {eg.summary}")
    else:
        eval_gate = {"passed": False, "gate_name": "evaluation", "checks": {}, "summary": "No metrics found"}
        logs.append("Eval Gate: SKIP — no metrics")

    # Drift gate
    if os.path.exists(config.baseline_stats_path):
        live_df = collector.get_predictions_since(config.drift_window_hours)
        dt = gate_cfg.get("drift", {})
        if len(live_df) >= dt.get("min_psi_samples", 30):
            report = check_drift(
                config.baseline_stats_path, live_df,
                p_threshold=dt.get("max_feature_drift_pvalue", 0.05),
                psi_threshold=dt.get("max_psi", 0.20),
                min_samples=dt.get("min_psi_samples", 30),
            )
            dg2 = run_drift_gate(dt, {"drifted_features": report.drifted_features, "overall_psi": report.overall_psi})
            drift_gate = {"passed": dg2.passed, "gate_name": dg2.gate_name, "checks": dg2.checks, "summary": dg2.summary}
            logs.append(f"Drift Gate: {'PASS' if dg2.passed else 'FAIL'} — {dg2.summary}")
        else:
            n = len(live_df)
            min_s = dt.get("min_psi_samples", 30)
            drift_gate = {"passed": True, "gate_name": "drift", "checks": {}, "summary": f"Insufficient data ({n}/{min_s} samples)"}
            logs.append(f"Drift Gate: SKIP — {n}/{min_s} samples")
    else:
        drift_gate = {"passed": False, "gate_name": "drift", "checks": {}, "summary": "No baseline stats"}
        logs.append("Drift Gate: SKIP — no baseline")

    return jsonify({
        "data_gate": data_gate,
        "eval_gate": eval_gate,
        "drift_gate": drift_gate,
        "logs": logs,
    })


# ---------------------------------------------------------------------------
# API: Configuration
# ---------------------------------------------------------------------------

@app.route("/api/config/gates", methods=["GET"])
def api_config_gates_get():
    return jsonify(load_gate_config(config.gates_config))


@app.route("/api/config/gates", methods=["PUT"])
def api_config_gates_put():
    body = request.get_json()
    if body is None:
        return jsonify({"error": "Invalid JSON"}), 400

    gates_path = config.gates_config
    with open(gates_path, "w") as f:
        yaml.dump(body, f, default_flow_style=False, sort_keys=False)

    global gate_cfg
    gate_cfg = load_gate_config(gates_path)

    return jsonify(gate_cfg)


@app.route("/api/config/app", methods=["GET"])
def api_config_app():
    return jsonify({
        "data_path": config.data_path,
        "model_dir": config.model_dir,
        "gates_config": config.gates_config,
        "experiment_name": config.experiment_name,
        "port": config.port,
        "log_level": config.log_level,
        "drift_check_interval": config.drift_check_interval,
        "drift_window_hours": config.drift_window_hours,
        "monitoring_retention_days": config.monitoring_retention_days,
        "mlflow_uri": config.mlflow_uri,
        "telemetry_db": config.telemetry_db,
        "alerts_log": config.alerts_log,
        "baseline_stats_path": config.baseline_stats_path,
    })


# ---------------------------------------------------------------------------
# API: Pipeline
# ---------------------------------------------------------------------------

@app.route("/api/pipeline/run", methods=["POST"])
def api_pipeline_run():
    global model, preprocessor, model_loaded

    logs = []

    class LogCapture:
        def __init__(self, logs_list):
            self.logs = logs_list
        def write(self, text):
            self.logs.append(text.strip())
        def flush(self):
            pass

    log_capture = LogCapture(logs)

    try:
        log_capture.write("=== Training Pipeline with Gates ===")
        log_capture.write(f"Data: {config.data_path}")

        config.ensure_dirs()
        df = load_data(config.data_path)

        data_gate_result = None
        eval_gate_result = None

        data_report = validate_data(df, **gate_cfg.get("data_validation", {}))
        log_capture.write(f"Data validation: {data_report.passed} ({len(data_report.warnings)} warnings)")

        dg = run_data_gate(gate_cfg.get("data_validation", {}), data_report)
        data_gate_result = {"passed": dg.passed, "gate_name": dg.gate_name, "checks": dg.checks, "summary": dg.summary}
        log_capture.write(f"Data Gate: {'PASS' if dg.passed else 'FAIL'} — {dg.summary}")

        if not dg.passed:
            log_capture.write("Training aborted due to data gate failure.")
            return jsonify({
                "success": False,
                "logs": logs,
                "data_gate": data_gate_result,
                "eval_gate": None,
                "drift_gate": None,
            })

        pipe_preprocessor = get_preprocessor()
        X, y, _ = build_features(df, pipe_preprocessor, fit=True)
        log_capture.write(f"Features built: {X.shape[1]} features, {X.shape[0]} samples")

        pipe_model, X_train, X_test, y_train, y_test = train_model(X, y, config)
        log_capture.write("Model training complete")

        metrics = evaluate_model(pipe_model, X_test, y_test, output_path=f"{config.model_dir}/metrics.json")
        log_capture.write(f"Accuracy: {metrics['accuracy']}, ROC AUC: {metrics['roc_auc']}")

        eg = run_eval_gate(gate_cfg.get("evaluation", {}), metrics)
        eval_gate_result = {"passed": eg.passed, "gate_name": eg.gate_name, "checks": eg.checks, "summary": eg.summary}
        log_capture.write(f"Eval Gate: {'PASS' if eg.passed else 'FAIL'} — {eg.summary}")

        model_save_path = f"{config.model_dir}/model.pkl"
        preprocessor_save_path = f"{config.model_dir}/preprocessor.pkl"
        joblib.dump(pipe_model, model_save_path)
        joblib.dump(pipe_preprocessor, preprocessor_save_path)
        log_capture.write(f"Model saved to {model_save_path}")
        log_capture.write(f"Preprocessor saved to {preprocessor_save_path}")

        baseline = {
            "numeric": {},
            "categorical": {},
            "n_samples": len(df),
        }
        for col in ["age", "tenure_months", "monthly_charges", "total_charges",
                     "avg_monthly_usage_hours", "late_payments_last_12m"]:
            baseline["numeric"][col] = {
                "mean": round(float(df[col].mean()), 2),
                "std": round(float(df[col].std()), 2),
                "p50": round(float(df[col].median()), 2),
                "p95": round(float(df[col].quantile(0.95)), 2),
            }
        for col in CAT_ORDERINGS:
            baseline["categorical"][col] = df[col].value_counts(normalize=True).to_dict()

        with open(config.baseline_stats_path, "w") as f:
            json.dump(baseline, f, indent=2)
        log_capture.write(f"Baseline stats saved to {config.baseline_stats_path}")

        # Reload model in serving context if possible
        try:
            model = joblib.load(model_save_path)
            preprocessor = joblib.load(preprocessor_save_path)
            model_loaded = True
            log_capture.write("Model reloaded into serving context")
        except Exception as e:
            log_capture.write(f"Warning: could not reload model: {e}")

        log_capture.write("=== Pipeline Complete ===")

        return jsonify({
            "success": True,
            "logs": logs,
            "data_gate": data_gate_result,
            "eval_gate": eval_gate_result,
            "drift_gate": None,
        })

    except Exception as e:
        logs.append(f"PIPELINE ERROR: {e}")
        return jsonify({
            "success": False,
            "logs": logs,
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = config.port
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=(config.log_level == "DEBUG"))
