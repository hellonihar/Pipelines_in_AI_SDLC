import json
import os
import tempfile

import numpy as np
import pandas as pd

from src.monitoring.collector import TelemetryCollector
from src.monitoring.drift import compute_psi, compute_categorical_psi
from src.monitoring.alert import AlertManager


class TestTelemetryCollector:
    def test_log_and_summary(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        collector = TelemetryCollector(db_path)
        collector.log_prediction(
            {"age": 35, "tenure_months": 12, "monthly_charges": 75.5,
             "total_charges": 906.0, "avg_monthly_usage_hours": 45,
             "late_payments_last_12m": 3, "contract_type": "Month-to-month",
             "payment_method": "Electronic check", "internet_service": "Fiber optic",
             "tech_support": "No"},
            prediction=1, confidence_churn=0.85, confidence_stay=0.15,
        )

        summary = collector.get_summary()
        assert summary["total_predictions"] == 1
        assert summary["churn_rate"] == 1.0

        os.unlink(db_path)

    def test_prune(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        collector = TelemetryCollector(db_path, retention_days=0)
        collector.prune()
        summary = collector.get_summary()
        assert summary["total_predictions"] == 0

        os.unlink(db_path)


class TestDrift:
    def test_compute_psi_identical(self):
        a = np.array([1.0] * 50 + [2.0] * 50)
        psi = compute_psi(a, a, bins=5)
        assert psi < 1e-6

    def test_compute_psi_different(self):
        a = np.array([1.0] * 50 + [2.0] * 50)
        b = np.array([1.5] * 50 + [2.5] * 50)
        psi = compute_psi(a, b, bins=5)
        assert psi > 0.01

    def test_categorical_psi_identical(self):
        psi = compute_categorical_psi({"A": 50, "B": 50}, {"A": 50, "B": 50})
        assert psi < 1e-6

    def test_categorical_psi_different(self):
        psi = compute_categorical_psi({"A": 100, "B": 0}, {"A": 0, "B": 100})
        assert psi > 1.0


class TestAlertManager:
    def test_write_alert(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        alert = AlertManager(log_path)
        alert.info("test", "info message")
        alert.warning("test", "warning message")

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert "INFO" in lines[0]
        assert "WARNING" in lines[1]

        os.unlink(log_path)
