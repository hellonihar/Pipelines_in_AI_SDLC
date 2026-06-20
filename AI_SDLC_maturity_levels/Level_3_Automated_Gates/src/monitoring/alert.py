from datetime import datetime, timezone


class AlertManager:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def _write(self, level: str, source: str, message: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {level} [{source}] {message}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        print(f"{level} [{source}] {message}")

    def info(self, source: str, message: str):
        self._write("INFO", source, message)

    def warning(self, source: str, message: str):
        self._write("WARNING", source, message)

    def critical(self, source: str, message: str):
        self._write("CRITICAL", source, message)

    def alert_from_gate(self, gate_result):
        level = "CRITICAL" if not gate_result.passed else "INFO"
        self._write(level, gate_result.gate_name, gate_result.summary)

    def alert_from_drift(self, drift_report):
        n = drift_report.n_drifted
        if n == 0:
            return
        level = "CRITICAL" if n > 2 else "WARNING"
        features = list(drift_report.drifted_features.keys())
        self._write(level, "drift", f"{n} features drifted: {features}. PSI: {drift_report.overall_psi:.4f}")
