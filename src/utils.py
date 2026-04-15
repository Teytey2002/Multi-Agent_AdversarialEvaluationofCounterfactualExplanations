import os
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model and return metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    return metrics


def save_model(model, filepath):
    """
    Save a trained sklearn pipeline with joblib.
    """
    joblib.dump(model, filepath)


def save_metrics(metrics, filepath):
    """
    Save metrics dictionary as JSON.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)