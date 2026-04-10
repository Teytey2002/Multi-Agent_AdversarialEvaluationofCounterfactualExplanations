import pandas as pd
import joblib
import dice_ml

from data_loader import load_adult_dataset


MODEL_PATH = "models/xgboost.joblib"
SAMPLE_PATH = "results/unfavorable_samples.csv"


def main():
    # =========================
    # Load model
    # =========================
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    # =========================
    # Load full dataset
    # =========================
    X, y = load_adult_dataset()

    data = X.copy()
    data["income"] = y  # DiCE veut une target dans le dataset

    # =========================
    # Load sample
    # =========================
    sample = pd.read_csv(SAMPLE_PATH)

    # Remove extra columns (DiCE veut seulement les features)
    sample = sample[X.columns]

    # =========================
    # Define DiCE data object
    # =========================
    d = dice_ml.Data(
        dataframe=data,
        continuous_features=[
            "age", "fnlwgt", "education-num",
            "capital-gain", "capital-loss", "hours-per-week"
        ],
        outcome_name="income"
    )

    # =========================
    # Model wrapper
    # =========================
    m = dice_ml.Model(
        model=model,
        backend="sklearn"
    )

    exp = dice_ml.Dice(d, m, method="random")

    # =========================
    # Features allowed to change
    # =========================
    features_to_vary = [
        "workclass",
        "education",
        "occupation",
        "hours-per-week",
        "capital-gain",
        "capital-loss"
    ]

    # =========================
    # Generate CF
    # =========================
    all_results = []

    for i in range(len(sample)):
        instance = sample.iloc[[i]]

        print("\n" + "=" * 60)
        print(f"Generating CF for instance {i}")
        print("=" * 60)

        cf = exp.generate_counterfactuals(
            instance,
            total_CFs=3,
            desired_class="opposite",
            features_to_vary=features_to_vary
        )

        df_cf = cf.cf_examples_list[0].final_cfs_df

        print("\nOriginal instance:")
        print(instance)

        print("\nCounterfactuals:")
        print(df_cf)

        df_cf["original_index"] = i
        all_results.append(df_cf)

    # =========================
    # Save results
    # =========================
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("results/counterfactuals.csv", index=False)

    print("\nSaved counterfactuals to results/counterfactuals.csv")


if __name__ == "__main__":
    main()