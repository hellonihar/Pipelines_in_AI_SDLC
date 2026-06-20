import pytest
from src.gates import run_data_gate, run_eval_gate, run_drift_gate


class TestDataGate:
    def test_passes_valid_report(self, sample_df):
        from src.data.validate import validate_data
        report = validate_data(sample_df)
        result = run_data_gate({
            "min_rows": 1, "max_null_fraction": 1.0,
            "max_anomaly_rate": 1.0, "max_novel_category_rate": 1.0,
        }, report)
        assert result.passed
        assert result.gate_name == "data_validation"

    def test_fails_on_too_few_rows(self, sample_df):
        from src.data.validate import validate_data
        report = validate_data(sample_df)
        result = run_data_gate({
            "min_rows": 100, "max_null_fraction": 1.0,
            "max_anomaly_rate": 1.0, "max_novel_category_rate": 1.0,
        }, report)
        assert not result.passed


class TestEvalGate:
    def test_passes_good_metrics(self):
        metrics = {"accuracy": 0.9, "precision": 0.85, "recall": 0.8, "f1_score": 0.82, "roc_auc": 0.88}
        result = run_eval_gate({
            "min_accuracy": 0.75, "min_precision": 0.70,
            "min_recall": 0.65, "min_f1": 0.70, "min_roc_auc": 0.75,
        }, metrics)
        assert result.passed

    def test_fails_below_threshold(self):
        metrics = {"accuracy": 0.5, "precision": 0.4, "recall": 0.3, "f1_score": 0.35, "roc_auc": 0.5}
        result = run_eval_gate({
            "min_accuracy": 0.75, "min_precision": 0.70,
            "min_recall": 0.65, "min_f1": 0.70, "min_roc_auc": 0.75,
        }, metrics)
        assert not result.passed

    def test_missing_metric_fails(self):
        result = run_eval_gate({"min_accuracy": 0.75}, {})
        assert not result.passed


class TestDriftGate:
    def test_passes_no_drift(self):
        result = run_drift_gate({"max_feature_drift_pvalue": 0.05, "max_psi": 0.20}, {
            "drifted_features": {},
            "overall_psi": 0.05,
        })
        assert result.passed

    def test_fails_on_drift(self, drift_report):
        result = run_drift_gate({"max_feature_drift_pvalue": 0.05, "max_psi": 0.20}, {
            "drifted_features": drift_report["drifted_features"],
            "overall_psi": drift_report["overall_psi"],
        })
        assert not result.passed
