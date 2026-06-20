import joblib

from src.config import Config
from src.data.load import load_data
from src.features.build import build_features, get_preprocessor
from src.evaluate import evaluate_model


def main():
    config = Config()

    print("=== Level 3: Evaluation Pipeline ===")
    print()

    df = load_data(config.data_path)

    preprocessor = joblib.load(f"{config.model_dir}/preprocessor.pkl")
    X, y, _ = build_features(df, preprocessor, fit=False)

    model = joblib.load(f"{config.model_dir}/model.pkl")

    metrics = evaluate_model(model, X, y, output_path=f"{config.model_dir}/metrics.json")
    print(f"\nAll metrics: {config.model_dir}/metrics.json")


if __name__ == "__main__":
    main()
