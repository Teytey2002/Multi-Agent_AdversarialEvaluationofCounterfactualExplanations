import os
import numpy as np
import pandas as pd
import joblib
import dice_ml

from data_loader import load_adult_dataset


MODEL_PATH = "models/logistic_regression.joblib"
SAMPLE_PATH = "results/unfavorable_samples.csv"
OUTPUT_PATH = "results/counterfactuals.csv"

# Nombre de CFs par individu
TOTAL_CFS = 4

# Classe désirée : 1 = >50K
DESIRED_CLASS = 1


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean potential missing markers and drop incomplete rows.
    This avoids generating CFs with NaN-like values.
    """
    df = df.replace("?", np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def get_actionable_features():
    """
    Features allowed to change.
    Based on the paper's feasibility/actionability spirit:
    - keep protected or clearly non-actionable features fixed
    - avoid education for now, because changing education without
      allowing age to evolve is causally questionable
    """
    return [
        "workclass",
        "occupation",
        "hours-per-week",
        "capital-gain",
        "capital-loss"
    ]


def get_permitted_range():
    """
    Box constraints inspired by the paper's user constraints idea.
    These ranges are intentionally conservative to avoid absurd CFs.
    """
    return {
        "hours-per-week": [20, 50],
        "capital-gain": [0, 5000],
        "capital-loss": [0, 5000]
    }


def build_dice_objects(model, data):
    """
    Build DiCE data/model/explainer objects.
    data : full dataset (including both features and target) needed to fit the DiCE data object.
    model : trained model with predict and predict_proba methods.
    exp : DiCE explainer object.
    """
    d = dice_ml.Data(
        dataframe=data,
        continuous_features=[
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
        ],
        outcome_name="income"
    )

    m = dice_ml.Model(
        model=model,
        backend="sklearn"
    )

    # 'genetic' is generally a better fit than 'random' when we want
    # a more controlled search with diversity/proximity trade-offs.
    exp = dice_ml.Dice(d, m, method="genetic")

    return d, m, exp


def generate_for_instance(exp, instance, features_to_vary, permitted_range):
    """
    Generate counterfactuals for one instance.
    """
    cf = exp.generate_counterfactuals(
        query_instances=instance,
        total_CFs=TOTAL_CFS,
        desired_class=DESIRED_CLASS,
        features_to_vary=features_to_vary,
        permitted_range=permitted_range,
        proximity_weight=1.0,
        diversity_weight=3.0,
        categorical_penalty=1.0,
        stopping_threshold=0.5,
        posthoc_sparsity_param=0.1,
        posthoc_sparsity_algorithm="linear",
        verbose=False
    )

    return cf


def format_results(original_row, cf_df, original_index):
    """
    Save both original and CFs in a single table for later evaluation.
    """
    rows = []

    original_out = original_row.copy()
    original_out["income"] = 0
    original_out["original_index"] = original_index
    original_out["row_type"] = "original"
    original_out["cf_rank"] = -1
    rows.append(original_out)

    if cf_df is not None and not cf_df.empty:
        for rank, (_, row) in enumerate(cf_df.iterrows()):
            out = row.copy()
            out["original_index"] = original_index
            out["row_type"] = "counterfactual"
            out["cf_rank"] = rank
            rows.append(out)

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

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
    data["income"] = y
    data = clean_dataframe(data)

    # =========================
    # Load unfavorable samples
    # =========================
    sample = pd.read_csv(SAMPLE_PATH)
    sample = sample.replace("?", np.nan)
    sample = sample.dropna().reset_index(drop=True)

    # Keep only feature columns
    sample = sample[X.columns]

    # Align with cleaned dataset schema
    sample = sample.dropna().reset_index(drop=True)

    # =========================
    # DiCE objects
    # =========================
    _, _, exp = build_dice_objects(model, data)

    # =========================
    # Constraints
    # =========================
    features_to_vary = get_actionable_features()
    permitted_range = get_permitted_range()

    print("\nFeatures allowed to vary:")
    print(features_to_vary)

    print("\nPermitted ranges:")
    print(permitted_range)

    # =========================
    # Generate CFs
    # =========================
    all_results = []

    for i in range(len(sample)):
        instance = sample.iloc[[i]].copy()

        print("\n" + "=" * 70)
        print(f"Generating CFs for instance {i}")
        print("=" * 70)
        print("Original instance:")
        print(instance.to_string(index=False))

        try:
            cf = generate_for_instance(
                exp=exp,
                instance=instance,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range
            )

            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is None or cf_df.empty:
                print("No valid counterfactual found.")
                result_block = format_results(
                    original_row=instance.iloc[0],
                    cf_df=None,
                    original_index=i
                )
            else:
                print("\nGenerated counterfactuals:")
                print(cf_df.to_string(index=False))

                result_block = format_results(
                    original_row=instance.iloc[0],
                    cf_df=cf_df,
                    original_index=i
                )

            all_results.append(result_block)

        except Exception as e:
            print(f"Error while generating CFs for instance {i}: {e}")

            # Save at least the original row so downstream scripts remain usable
            result_block = format_results(
                original_row=instance.iloc[0],
                cf_df=None,
                original_index=i
            )
            all_results.append(result_block)

    # =========================
    # Save all
    # =========================
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved improved counterfactuals to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()