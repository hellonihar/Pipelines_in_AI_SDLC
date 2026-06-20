import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_model


class TestEvaluateModel:
    def test_returns_all_expected_metrics(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)

        expected_keys = {
            "accuracy", "precision", "recall",
            "f1_score", "roc_auc", "n_samples",
            "n_positives", "n_negatives", "confusion_matrix",
        }
        assert set(metrics.keys()) == expected_keys

    def test_metrics_are_in_range(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)

        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_confusion_matrix_is_2x2(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
