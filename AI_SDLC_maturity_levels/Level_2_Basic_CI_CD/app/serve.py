import os
import sys
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

from src.config import Config
from src.features.build import CAT_ORDERINGS

config = Config()
app = Flask(__name__)

model_path = os.getenv("MODEL_PATH", f"{config.model_dir}/model.pkl")
preprocessor_path = os.getenv("PREPROCESSOR_PATH", f"{config.model_dir}/preprocessor.pkl")

if not os.path.exists(model_path):
    print(f"FATAL: Model not found at {model_path}", file=sys.stderr)
    sys.exit(1)

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

feature_names = (
    ["age", "tenure_months", "monthly_charges", "total_charges",
     "avg_monthly_usage_hours", "late_payments_last_12m"]
    + list(CAT_ORDERINGS.keys())
)

print(f"Model loaded from {model_path}")
print(f"Expected features: {feature_names}")


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

    return jsonify({"predictions": results})


if __name__ == "__main__":
    port = config.port
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=(config.log_level == "DEBUG"))
