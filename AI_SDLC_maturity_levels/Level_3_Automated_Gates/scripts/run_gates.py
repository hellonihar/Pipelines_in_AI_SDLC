import argparse

import joblib

from src.config import Config
from src.data.load import load_data
from src.features.build import build_features, get_preprocessor, CAT_ORDERINGS
from src.evaluate import evaluate_model
from src.gates import load_gate_config, run_data_gate, run_eval_gate


def main():
    parser = argparse.ArgumentParser(description="Run quality gates on a trained model")
    parser.add_argument("--model-dir", help="Directory containing model.pkl")
    parser.add_argument("--config", help="Path to gates.yaml")
    parser.add_argument("--data", help="Path to evaluation dataset (defaults to training data)")
    args = parser.parse_args()

    config = Config()
    model_dir = args.model_dir or config.model_dir

    gate_path = args.config or config.gates_config
    gate_cfg = load_gate_config(gate_path)

    data_path = args.data or config.data_path
    df = load_data(data_path)

    preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
    X, y, _ = build_features(df, preprocessor, fit=False)

    model = joblib.load(f"{model_dir}/model.pkl")
    metrics = evaluate_model(model, X, y)

    print(f"\n=== Running Gates ===")
    all_passed = True

    eval_gate = run_eval_gate(gate_cfg.get("evaluation", {}), metrics)
    print(f"[{'PASS' if eval_gate.passed else 'FAIL'}] {eval_gate.gate_name}: {eval_gate.summary}")
    if not eval_gate.passed:
        all_passed = False

    print(f"\nOverall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
