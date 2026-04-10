import pandas as pd
import numpy as np


def compute_sparsity(original, cf):
    return (original != cf).sum()


def compute_proximity(original, cf, numeric_cols):
    return np.linalg.norm(original[numeric_cols].values - cf[numeric_cols].values)


def main():
    df = pd.read_csv("results/counterfactuals.csv")

    # Détection des colonnes numériques (une seule fois)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # On enlève les colonnes non pertinentes
    numeric_cols = [col for col in numeric_cols if col not in ["original_index", "income"]]

    results = []

    for idx in df["original_index"].unique():
        group = df[df["original_index"] == idx]

        # ⚠️ IMPORTANT : original = première ligne
        original = group.iloc[0]

        for _, cf in group.iterrows():
            sparsity = compute_sparsity(original, cf)
            proximity = compute_proximity(original, cf, numeric_cols)

            results.append({
                "original_index": idx,
                "sparsity": sparsity,
                "proximity": proximity
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/cf_metrics.csv", index=False)

    print("Saved metrics to results/cf_metrics.csv")

if __name__ == "__main__":
    main()