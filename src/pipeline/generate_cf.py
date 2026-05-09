import os
import json
import numpy as np
import pandas as pd
import joblib
import dice_ml

from pipeline.data_loader import load_adult_dataset
from policy.feature_policy import (
    ACTIONABLE_FEATURES,
    CONTINUOUS_FEATURES,
    DICE_DEFAULT_GENETIC_KWARGS,
    POLICY_NAME,
    build_permitted_range,
    generation_policy_metadata,
    select_model_features,
    sync_education_label,
    sync_education_labels,
)


MODEL_PATH = "models/logistic_regression.joblib"
SAMPLE_PATH = "results/unfavorable_samples.csv"
OUTPUT_PATH = "results/counterfactuals.csv"
POLICY_OUTPUT_PATH = "results/generation_policy.json"

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
    Features allowed to change under the current long-term recourse policy.

    ``education`` is excluded from training and generation because it duplicates
    ``education-num``.  ``age`` and ``education-num`` may increase, but their
    causal consistency is validated by deterministic heuristics after generation.
    """
    return list(ACTIONABLE_FEATURES)


def get_permitted_range(data: pd.DataFrame, instance: pd.DataFrame):
    """
    Build per-instance box constraints from empirical dataset percentiles.
    """
    return build_permitted_range(data, instance.iloc[0])


def build_dice_objects(model, data):
    """
    Build DiCE data/model/explainer objects.
    data : full dataset (including both features and target) needed to fit the DiCE data object.
    model : trained model with predict and predict_proba methods.
    exp : DiCE explainer object.
    """
    # fnlwgt is a census sampling weight with ~28 K unique values.
    # It is NOT in features_to_vary so DiCE will never mutate it,
    # but it MUST be declared continuous; otherwise DiCE treats it
    # as categorical and the genetic search space explodes.
    # A proper removal requires retraining the model without fnlwgt.
    d = dice_ml.Data(
        dataframe=data,
        continuous_features=[
            *CONTINUOUS_FEATURES
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
        **DICE_DEFAULT_GENETIC_KWARGS,
        verbose=False
    )

    return cf


def format_results(original_row, cf_df, original_index, model, feature_cols):
    """
    Save both original and CFs in a single table for later evaluation.
    Includes cf_confidence: model's predicted probability for class 1.
    """
    rows = []

    original_out = sync_education_label(original_row.copy())
    original_out["income"] = 0
    original_out["original_index"] = original_index
    original_out["row_type"] = "original"
    original_out["cf_rank"] = -1
    # confidence the model assigns to the original (class 1 proba)
    orig_df = select_model_features(pd.DataFrame([original_row[feature_cols]]))
    original_out["cf_confidence"] = float(model.predict_proba(orig_df)[:, 1][0])
    rows.append(original_out)

    if cf_df is not None and not cf_df.empty:
        cf_df = sync_education_labels(cf_df)
        for rank, (_, row) in enumerate(cf_df.iterrows()):
            out = row.copy()
            out["original_index"] = original_index
            out["row_type"] = "counterfactual"
            out["cf_rank"] = rank
            # confidence the model assigns to this CF (class 1 proba)
            cf_input = select_model_features(pd.DataFrame([row[feature_cols]]))
            out["cf_confidence"] = float(model.predict_proba(cf_input)[:, 1][0])
            rows.append(out)

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)
    policy_metadata = generation_policy_metadata()
    policy_metadata["total_cfs"] = TOTAL_CFS
    policy_metadata["desired_class"] = DESIRED_CLASS

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

    # Feature columns (needed for predict_proba on CFs)
    feature_cols = X.columns.tolist()

    # =========================
    # Constraints
    # =========================
    features_to_vary = get_actionable_features()

    print(f"\nGeneration policy: {POLICY_NAME}")
    print("\nFeatures allowed to vary:")
    print(features_to_vary)
    print("\nDiCE genetic parameters:")
    print(DICE_DEFAULT_GENETIC_KWARGS)

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
            permitted_range = get_permitted_range(data, instance)
            policy_metadata.setdefault("per_instance_permitted_range", {})[str(i)] = permitted_range

            print("\nPermitted ranges:")
            print(permitted_range)

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
                    original_index=i,
                    model=model,
                    feature_cols=feature_cols
                )
            else:
                cf_df = sync_education_labels(cf_df)
                print("\nGenerated counterfactuals:")
                print(cf_df.to_string(index=False))

                result_block = format_results(
                    original_row=instance.iloc[0],
                    cf_df=cf_df,
                    original_index=i,
                    model=model,
                    feature_cols=feature_cols
                )

            all_results.append(result_block)

        except Exception as e:
            print(f"Error while generating CFs for instance {i}: {e}")

            # Save at least the original row so downstream scripts remain usable
            result_block = format_results(
                original_row=instance.iloc[0],
                cf_df=None,
                original_index=i,
                model=model,
                feature_cols=feature_cols
            )
            all_results.append(result_block)

    # =========================
    # Save all
    # =========================
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    with open(POLICY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(policy_metadata, f, indent=2)

    print(f"\nSaved improved counterfactuals to: {OUTPUT_PATH}")
    print(f"Saved generation policy metadata to: {POLICY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
