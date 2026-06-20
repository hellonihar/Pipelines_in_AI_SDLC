import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)


def evaluate_model(model, X_test, y_test, output_path: str | None = None) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_samples": int(len(y_test)),
        "n_positives": int(np.sum(y_test)),
        "n_negatives": int(len(y_test) - np.sum(y_test)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    for name, value in metrics.items():
        if name != "confusion_matrix":
            print(f"{name}: {value}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")

    return metrics
