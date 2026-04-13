import joblib
import pandas as pd

from data_loader import load_adult_dataset


MODEL_PATH = "models/logistic_regression.joblib"


def main():
    # Load model
    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    # Load data
    X, y = load_adult_dataset()

    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df = X.copy()
    df["prediction"] = preds
    df["proba"] = probs
    df["true_label"] = y

    # Filter "unfavorable" cases (prediction = 0)
    unfavorable = df[df["prediction"] == 0]

    print("\nNumber of unfavorable cases:", len(unfavorable))

    # Take a few examples
    sample = unfavorable.sample(5, random_state=42)

    print("\nSample of unfavorable individuals:")
    print(sample)

    # Save for next step
    sample.to_csv("results/unfavorable_samples.csv", index=False)

    print("\nSaved sample to results/unfavorable_samples.csv")


if __name__ == "__main__":
    main()