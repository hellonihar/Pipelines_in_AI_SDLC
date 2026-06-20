import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_path: str = field(default_factory=lambda: os.getenv(
        "DATA_PATH", str(Path(__file__).resolve().parent.parent / "data" / "customer_data.csv")
    ))
    model_dir: str = field(default_factory=lambda: os.getenv(
        "MODEL_DIR", str(Path(__file__).resolve().parent.parent / "models")
    ))
    gates_config: str = field(default_factory=lambda: os.getenv(
        "GATES_CONFIG", str(Path(__file__).resolve().parent.parent / "config" / "gates.yaml")
    ))
    experiment_name: str = os.getenv("EXPERIMENT_NAME", "churn-prediction")
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    n_estimators: int = int(os.getenv("N_ESTIMATORS", "100"))
    max_depth: int = int(os.getenv("MAX_DEPTH", "10"))
    port: int = int(os.getenv("PORT", "8080"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    drift_check_interval: int = int(os.getenv("DRIFT_CHECK_INTERVAL", "3600"))
    drift_window_hours: int = int(os.getenv("DRIFT_WINDOW_HOURS", "24"))
    monitoring_retention_days: int = int(os.getenv("MONITORING_RETENTION_DAYS", "90"))

    _mlflow_path: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent / "mlflow_data"
    ))
    _monitoring_dir: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent.parent / "monitoring"
    ))

    @property
    def mlflow_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + self._mlflow_path.replace("\\", "/") + "/mlflow.db")

    @property
    def telemetry_db(self) -> str:
        return os.path.join(self._monitoring_dir, "telemetry.db")

    @property
    def alerts_log(self) -> str:
        return os.path.join(self._monitoring_dir, "alerts.log")

    @property
    def baseline_stats_path(self) -> str:
        return os.path.join(self._monitoring_dir, "reference", "baseline_stats.json")

    def ensure_dirs(self):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self._mlflow_path).mkdir(parents=True, exist_ok=True)
        Path(self._monitoring_dir).mkdir(parents=True, exist_ok=True)
        Path(self._monitoring_dir, "reference").mkdir(parents=True, exist_ok=True)
