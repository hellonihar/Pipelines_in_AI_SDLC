import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import Config


def train_model(
    X, y, config: Config,
    log_to_mlflow: bool = True,
) -> tuple[object, object, object, object]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_seed,
    )

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed,
    )

    if log_to_mlflow:
        mlflow.set_tracking_uri(config.mlflow_uri)
        mlflow.set_experiment(config.experiment_name)

        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "random_seed": config.random_seed,
                "test_size": config.test_size,
                "model_type": "RandomForestClassifier",
            })

            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            mlflow.log_metrics({
                "train_accuracy": train_score,
                "test_accuracy": test_score,
            })

            mlflow.sklearn.log_model(model, "model")

            print(f"Train accuracy: {train_score:.4f}")
            print(f"Test accuracy:  {test_score:.4f}")
            print(f"MLflow run ID:  {mlflow.active_run().info.run_id}")
    else:
        model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
