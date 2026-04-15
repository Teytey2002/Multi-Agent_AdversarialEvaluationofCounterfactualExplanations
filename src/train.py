import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_adult_dataset
from preprocessing import build_preprocessor
from models import get_model
from utils import ensure_dir, evaluate_model, save_model, save_metrics


RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS_DIR = "models"
RESULTS_DIR = "results"


def train():
    """Train the Logistic Regression pipeline and save the model + metrics."""
    # Create output folders
    ensure_dir(MODELS_DIR)
    ensure_dir(RESULTS_DIR)

    # Load data
    X, y = load_adult_dataset()

    print(f"Dataset loaded: X shape = {X.shape}, y shape = {y.shape}")
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Build preprocessor once
    preprocessor = build_preprocessor(X)

    # Get model
    model_name, classifier = get_model(random_state=RANDOM_STATE)

    print("\n" + "=" * 70)
    print(f"Training model: {model_name}")
    print("=" * 70)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    save_model(pipeline, model_path)

    # Save detailed metrics
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    save_metrics(metrics, metrics_path)

    print("\n" + "=" * 70)
    print("Training completed.")
    print(f"Model saved to : {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()