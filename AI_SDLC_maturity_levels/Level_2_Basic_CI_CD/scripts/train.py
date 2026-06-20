import joblib

from src.config import Config
from src.data.load import load_data
from src.features.build import build_features, get_preprocessor
from src.models.train import train_model


def main():
    config = Config()
    config.ensure_dirs()

    print("=== Level 2: Training Pipeline ===")
    print(f"Data: {config.data_path}")
    print(f"Seed: {config.random_seed}")
    print()

    df = load_data(config.data_path)

    preprocessor = get_preprocessor()
    X, y, _ = build_features(df, preprocessor, fit=True)

    model, X_train, X_test, y_train, y_test = train_model(X, y, config)

    model_path = f"{config.model_dir}/model.pkl"
    preprocessor_path = f"{config.model_dir}/preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Model saved to:       {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")


if __name__ == "__main__":
    main()
