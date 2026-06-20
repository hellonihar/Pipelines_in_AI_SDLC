# serve_model.py — Level 1 Manual Model Serving
#
# A minimal Flask app that serves the churn model.
# No health checks, no metrics, no graceful shutdown, no auth.
# Typical of Level 1 — built once and forgotten.

import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# === Hardcoded model path ===
# Change this line every time you deploy a new model.
# No environment variables, no config files.
MODEL_PATH = "../models/churn_model_2026-06-20.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"FATAL: Model not found at {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

model = joblib.load(MODEL_PATH)
feature_names = [
    "age", "tenure_months", "monthly_charges", "total_charges",
    "contract_type", "payment_method", "internet_service",
    "tech_support", "avg_monthly_usage_hours", "late_payments_last_12m"
]
print(f"Model loaded from {MODEL_PATH}")
print(f"Expected features: {feature_names}")


@app.route("/predict", methods=["POST"])
def predict():
    # No input validation — malformed JSON crashes the app
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    # Supports single dict or list of dicts — fragile branching
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        return jsonify({"error": "Expected object or array"}), 400

    # Missing columns → KeyError → 500
    try:
        X = df[feature_names]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    preds = model.predict(X).tolist()
    probs = model.predict_proba(X).tolist()

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append({
            "prediction": int(pred),
            "label": "CHURN" if pred == 1 else "STAY",
            "confidence_stay": round(prob[0], 4),
            "confidence_churn": round(prob[1], 4)
        })

    return jsonify({"predictions": results})


# === Raw app.run — no gunicorn, no workers, no reverse proxy ===
# If this crashes, it stays down until someone manually restarts it.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}...")
    print("WARNING: Running in debug mode. Do not use in production.")
    app.run(host="0.0.0.0", port=port, debug=True)
