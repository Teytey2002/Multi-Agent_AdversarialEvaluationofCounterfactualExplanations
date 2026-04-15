"""
explore_data.py — Catalogue every feature's possible values in the Adult
Income dataset.

Produces ``results/feature_catalog.json`` — a reference artifact that helps
inform the issue taxonomy and agent prompt design.

Usage (from repo root):
    $env:PYTHONPATH="src"; python src/explore_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from data_loader import load_adult_dataset


RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "feature_catalog.json"

# fnlwgt is a census sampling weight with ~28 K unique values — not
# meaningful for taxonomy design, so we skip it.
SKIP_FEATURES = {"fnlwgt"}


def _describe_categorical(series) -> dict:
    """Summarise a categorical feature."""
    counts = series.value_counts(dropna=False).sort_index()
    return {
        "type": "categorical",
        "n_unique": int(counts.shape[0]),
        "values": {str(k): int(v) for k, v in counts.items()},
    }


def _describe_numerical(series) -> dict:
    """Summarise a numerical feature."""
    clean = series.dropna()
    return {
        "type": "numerical",
        "n_unique": int(clean.nunique()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": round(float(clean.mean()), 2),
        "median": float(clean.median()),
        "std": round(float(clean.std()), 2),
    }


def build_feature_catalog() -> dict:
    """Load the dataset and return a per-feature catalog dict."""
    X, y = load_adult_dataset()

    catalog: dict[str, dict] = {}

    for col in X.columns:
        if col in SKIP_FEATURES:
            continue

        if X[col].dtype in ("object", "category"):
            catalog[col] = _describe_categorical(X[col])
        else:
            catalog[col] = _describe_numerical(X[col])

    # Also summarise the target distribution
    catalog["_target (income)"] = {
        "type": "binary",
        "mapping": {"<=50K": 0, ">50K": 1},
        "distribution": {
            "<=50K": int((y == 0).sum()),
            ">50K": int((y == 1).sum()),
        },
    }

    return catalog


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    catalog = build_feature_catalog()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Feature catalog saved → {OUTPUT_PATH}")
    print(f"Features catalogued: {len(catalog) - 1} (+ target)\n")

    for name, info in catalog.items():
        if name.startswith("_"):
            continue
        if info["type"] == "categorical":
            values = list(info["values"].keys())
            print(f"  {name:20s}  categorical  {info['n_unique']:>3} values  {values}")
        else:
            print(f"  {name:20s}  numerical    range [{info['min']}, {info['max']}]  "
                  f"mean={info['mean']}  median={info['median']}")


if __name__ == "__main__":
    main()
