import mlflow
from src.config import Config


def register_model(config: Config, run_id: str, stage: str = "staging"):
    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment(config.experiment_name)

    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "stage", stage)

    model_uri = f"runs:/{run_id}/model"
    model_name = config.experiment_name

    try:
        registered = mlflow.register_model(model_uri, model_name)
        client.set_registered_model_alias(model_name, stage, registered.version)
        print(f"Model registered as '{model_name}' version {registered.version} (stage: {stage})")
    except mlflow.MlflowException as e:
        print(f"Model registration skipped (MLflow tracking server may not support registry): {e}")


def get_model_source(config: Config, stage: str = "staging") -> str | None:
    mlflow.set_tracking_uri(config.mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        latest = client.get_latest_versions(config.experiment_name, stages=[stage])
        if latest:
            return f"runs:/{latest[0].run_id}/model"
    except Exception:
        pass
    return None
