import joblib
import numpy as np
import pandas as pd

from pipeline.data_loader import load_adult_dataset
from policy.feature_policy import select_model_features


MODEL_PATH = "models/logistic_regression.joblib"

# Number of unfavorable individuals to sample for CF generation.
# Increase for larger experiments; keep small for quick iteration.
SAMPLE_SIZE = 10

RANDOM_STATE = 42


def main():
    # Load model
    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    # Load data
    X, y = load_adult_dataset()
    X_model = select_model_features(X)

    # Predict
    preds = model.predict(X_model)
    probs = model.predict_proba(X_model)[:, 1]

    df = X.copy()
    df["prediction"] = preds
    df["proba"] = probs
    df["true_label"] = y

    # Flag false negatives (model says <=50K but truth is >50K)
    df["is_false_negative"] = ((df["prediction"] == 0) & (df["true_label"] == 1))

    # Filter "unfavorable" cases (prediction = 0)
    unfavorable = df[df["prediction"] == 0].copy()

    # Drop rows with missing values so generate_cf.py receives clean data
    unfavorable = unfavorable.replace("?", np.nan).dropna()

    print(f"\nUnfavorable cases (total): {len(df[df['prediction'] == 0])}")
    print(f"Unfavorable cases (after dropping NaN): {len(unfavorable)}")

    # Sample
    sample = unfavorable.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    print(f"\nSampled {SAMPLE_SIZE} unfavorable individuals:")
    print(sample)

    # Save for next step
    sample.to_csv("results/unfavorable_samples.csv", index=False)

    print("\nSaved sample to results/unfavorable_samples.csv")


if __name__ == "__main__":
    main()
