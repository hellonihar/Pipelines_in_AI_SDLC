import json
import joblib

from src.config import Config
from src.data.load import load_data
from src.data.validate import validate_data
from src.features.build import build_features, get_preprocessor, CAT_ORDERINGS
from src.models.train import train_model
from src.models.registry import register_model
from src.evaluate import evaluate_model
from src.gates import load_gate_config, run_data_gate, run_eval_gate


def main():
    config = Config()
    config.ensure_dirs()

    print("=== Level 3: Training Pipeline with Gates ===")
    print(f"Data: {config.data_path}")
    print(f"Gates config: {config.gates_config}")
    print()

    df = load_data(config.data_path)

    gate_cfg = load_gate_config(config.gates_config)
    data_report = validate_data(df, **gate_cfg.get("data_validation", {}))

    data_gate = run_data_gate(gate_cfg.get("data_validation", {}), data_report)
    if not data_gate.passed:
        print(f"\nDATA GATE FAILED: {data_gate.summary}")
        print("Training aborted.")
        return
    print(f"Data gate passed: {data_gate.summary}\n")

    preprocessor = get_preprocessor()
    X, y, _ = build_features(df, preprocessor, fit=True)

    model, X_train, X_test, y_train, y_test = train_model(X, y, config)

    metrics = evaluate_model(model, X_test, y_test, output_path=f"{config.model_dir}/metrics.json")

    eval_gate = run_eval_gate(gate_cfg.get("evaluation", {}), metrics)
    if not eval_gate.passed:
        print(f"\nEVAL GATE FAILED: {eval_gate.summary}")
        print("Model marked as 'rejected' in MLflow.")
        stage = "rejected"
    else:
        print(f"Eval gate passed: {eval_gate.summary}")
        stage = "staging"

    model_path = f"{config.model_dir}/model.pkl"
    preprocessor_path = f"{config.model_dir}/preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Model saved to:       {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")

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
    print(f"Baseline stats saved to: {config.baseline_stats_path}")


if __name__ == "__main__":
    main()
