import numpy as np
import pandas as pd

from data_loader import load_adult_dataset


CF_PATH = "results/counterfactuals.csv"
OUTPUT_PER_INSTANCE = "results/cf_metrics_per_instance.csv"
OUTPUT_GLOBAL = "results/cf_metrics_global.csv"


def get_feature_types(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return continuous_cols, categorical_cols


def compute_mad_values(X: pd.DataFrame, continuous_cols):
    """
    Compute Median Absolute Deviation for each continuous feature.

    When MAD = 0 (e.g. capital-gain where ~90 % of values are 0)
    we fall back to the standard deviation so that the normalised
    distance remains on a sensible scale.  If both MAD and std are
    zero the feature is constant and we use 1.0 (distances will be 0
    anyway).
    """
    mad = {}
    for col in continuous_cols:
        median = X[col].median()
        mad_val = np.median(np.abs(X[col] - median))

        if mad_val == 0 or pd.isna(mad_val):
            std_val = X[col].std()
            mad_val = std_val if (std_val > 0 and not pd.isna(std_val)) else 1.0

        mad[col] = mad_val
    return mad


def dist_cont(a: pd.Series, b: pd.Series, continuous_cols, mad_dict):
    if len(continuous_cols) == 0:
        return 0.0

    values = []
    for col in continuous_cols:
        values.append(abs(a[col] - b[col]) / mad_dict[col])

    return float(np.mean(values))


def dist_cat(a: pd.Series, b: pd.Series, categorical_cols):
    if len(categorical_cols) == 0:
        return 0.0

    diffs = [(a[col] != b[col]) for col in categorical_cols]
    return float(np.mean(diffs))


def sparsity_score(a: pd.Series, b: pd.Series, all_feature_cols):
    """
    Paper-style sparsity:
    1 - (#changed_features / total_features)
    Higher is better.
    """
    changed = sum(a[col] != b[col] for col in all_feature_cols)
    d = len(all_feature_cols)
    return 1.0 - (changed / d)


def count_diversity(cf_rows: pd.DataFrame, all_feature_cols):
    k = len(cf_rows)
    if k < 2:
        return 0.0

    pair_scores = []
    for i in range(k - 1):
        for j in range(i + 1, k):
            a = cf_rows.iloc[i]
            b = cf_rows.iloc[j]
            changed = sum(a[col] != b[col] for col in all_feature_cols)
            pair_scores.append(changed / len(all_feature_cols))

    return float(np.mean(pair_scores))


def pairwise_diversity(cf_rows: pd.DataFrame, continuous_cols, categorical_cols, mad_dict):
    k = len(cf_rows)
    if k < 2:
        return 0.0, 0.0

    cont_scores = []
    cat_scores = []

    for i in range(k - 1):
        for j in range(i + 1, k):
            a = cf_rows.iloc[i]
            b = cf_rows.iloc[j]

            cont_scores.append(dist_cont(a, b, continuous_cols, mad_dict))
            cat_scores.append(dist_cat(a, b, categorical_cols))

    return float(np.mean(cont_scores)), float(np.mean(cat_scores))


def compute_validity(cf_rows: pd.DataFrame):
    """
    Paper-style validity:
    fraction of requested CFs that are unique and valid.
    Here, we assume valid CFs are those with income == 1.
    """
    if len(cf_rows) == 0:
        return 0.0

    unique_cf_rows = cf_rows.drop_duplicates()
    valid_unique = unique_cf_rows[unique_cf_rows["income"] == 1]

    return float(len(valid_unique) / len(cf_rows))


def main():
    # dataset original pour types + MAD
    X, _ = load_adult_dataset()
    continuous_cols, categorical_cols = get_feature_types(X)
    mad_dict = compute_mad_values(X, continuous_cols)

    all_feature_cols = X.columns.tolist()

    df = pd.read_csv(CF_PATH)

    per_instance_results = []

    for idx in sorted(df["original_index"].unique()):
        group = df[df["original_index"] == idx].copy()

        original_rows = group[group["row_type"] == "original"]
        cf_rows = group[group["row_type"] == "counterfactual"].copy()

        if len(original_rows) != 1:
            print(f"Skipping original_index={idx}: expected exactly 1 original row.")
            continue

        original = original_rows.iloc[0]

        # garder uniquement les colonnes utiles pour comparaison
        cf_rows_features = cf_rows[all_feature_cols + ["income"]].drop_duplicates()
        cf_rows_only_features = cf_rows_features[all_feature_cols]

        # Validity
        validity = compute_validity(cf_rows_features)

        # Proximity
        if len(cf_rows_only_features) > 0:
            cont_prox_values = []
            cat_prox_values = []
            sparsity_values = []

            for _, cf in cf_rows_only_features.iterrows():
                cont_dist = dist_cont(original, cf, continuous_cols, mad_dict)
                cat_dist = dist_cat(original, cf, categorical_cols)

                cont_prox_values.append(-cont_dist)      # paper convention
                cat_prox_values.append(1.0 - cat_dist)  # paper convention
                sparsity_values.append(sparsity_score(original, cf, all_feature_cols))

            continuous_proximity = float(np.mean(cont_prox_values))
            categorical_proximity = float(np.mean(cat_prox_values))
            sparsity = float(np.mean(sparsity_values))
        else:
            continuous_proximity = 0.0
            categorical_proximity = 0.0
            sparsity = 0.0

        # Diversity
        continuous_diversity, categorical_diversity = pairwise_diversity(
            cf_rows_only_features,
            continuous_cols,
            categorical_cols,
            mad_dict
        )

        count_div = count_diversity(cf_rows_only_features, all_feature_cols)

        per_instance_results.append({
            "original_index": idx,
            "n_counterfactuals": len(cf_rows),
            "n_unique_counterfactuals": len(cf_rows_features),
            "validity": validity,
            "continuous_proximity": continuous_proximity,
            "categorical_proximity": categorical_proximity,
            "sparsity": sparsity,
            "continuous_diversity": continuous_diversity,
            "categorical_diversity": categorical_diversity,
            "count_diversity": count_div
        })

    results_df = pd.DataFrame(per_instance_results)
    results_df.to_csv(OUTPUT_PER_INSTANCE, index=False)

    global_results = pd.DataFrame([{
        "validity_mean": results_df["validity"].mean(),
        "continuous_proximity_mean": results_df["continuous_proximity"].mean(),
        "categorical_proximity_mean": results_df["categorical_proximity"].mean(),
        "sparsity_mean": results_df["sparsity"].mean(),
        "continuous_diversity_mean": results_df["continuous_diversity"].mean(),
        "categorical_diversity_mean": results_df["categorical_diversity"].mean(),
        "count_diversity_mean": results_df["count_diversity"].mean()
    }])

    global_results.to_csv(OUTPUT_GLOBAL, index=False)

    print(f"Saved per-instance metrics to: {OUTPUT_PER_INSTANCE}")
    print(f"Saved global metrics to: {OUTPUT_GLOBAL}")


if __name__ == "__main__":
    main()