import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta

import pandas as pd


class TelemetryCollector:
    def __init__(self, db_path: str, retention_days: int = 90):
        self.db_path = db_path
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    age REAL, tenure_months REAL,
                    monthly_charges REAL, total_charges REAL,
                    avg_monthly_usage_hours REAL,
                    late_payments_last_12m REAL,
                    contract_type TEXT, payment_method TEXT,
                    internet_service TEXT, tech_support TEXT,
                    prediction INTEGER,
                    confidence_churn REAL,
                    confidence_stay REAL
                )
            """)
            conn.commit()
            conn.close()

    def log_prediction(self, features: dict, prediction: int, confidence_churn: float, confidence_stay: float):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **features,
            "prediction": prediction,
            "confidence_churn": confidence_churn,
            "confidence_stay": confidence_stay,
        }
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO predictions (
                    timestamp, age, tenure_months,
                    monthly_charges, total_charges,
                    avg_monthly_usage_hours, late_payments_last_12m,
                    contract_type, payment_method,
                    internet_service, tech_support,
                    prediction, confidence_churn, confidence_stay
                ) VALUES (
                    :timestamp, :age, :tenure_months,
                    :monthly_charges, :total_charges,
                    :avg_monthly_usage_hours, :late_payments_last_12m,
                    :contract_type, :payment_method,
                    :internet_service, :tech_support,
                    :prediction, :confidence_churn, :confidence_stay
                )
            """, row)
            conn.commit()
            conn.close()

    def get_predictions_since(self, hours: int) -> pd.DataFrame:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                "SELECT * FROM predictions WHERE timestamp >= ? ORDER BY timestamp",
                conn, params=(cutoff.isoformat(),),
            )
            conn.close()
        return df

    def get_summary(self) -> dict:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute("SELECT COUNT(*), SUM(prediction), AVG(confidence_churn) FROM predictions")
            total, churn_sum, avg_conf = cur.fetchone()
            cur = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM predictions")
            t_min, t_max = cur.fetchone()
            conn.close()
        return {
            "total_predictions": total or 0,
            "churn_rate": round(churn_sum / total, 4) if total else 0.0,
            "avg_confidence_churn": round(avg_conf, 4) if avg_conf else 0.0,
            "time_range": {"from": t_min, "to": t_max},
        }

    def prune(self):
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff.isoformat(),))
            conn.commit()
            conn.close()
