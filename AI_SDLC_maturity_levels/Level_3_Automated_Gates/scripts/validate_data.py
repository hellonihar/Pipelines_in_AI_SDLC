import argparse

import pandas as pd

from src.config import Config
from src.data.validate import validate_data
from src.gates import load_gate_config, run_data_gate
from src.features.build import CAT_ORDERINGS


def main():
    parser = argparse.ArgumentParser(description="Validate a CSV dataset")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--config", help="Path to gates.yaml")
    args = parser.parse_args()

    config = Config()
    gate_path = args.config or config.gates_config
    gate_cfg = load_gate_config(gate_path).get("data_validation", {})

    print(f"Validating: {args.data}")
    print(f"Gates config: {gate_path}")
    print()

    df = pd.read_csv(args.data)
    report = validate_data(df, **gate_cfg)

    gate = run_data_gate(gate_cfg, report)
    print(f"\nGate result: {'PASSED' if gate.passed else 'FAILED'}")
    print(f"Summary: {gate.summary}")

    return 0 if gate.passed else 1


if __name__ == "__main__":
    exit(main())
