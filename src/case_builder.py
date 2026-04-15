"""
Bridge layer — converts Theo's ML pipeline outputs into structured
case dicts that the AutoGen multi-agent debate can consume directly.

Each case corresponds to one unfavorable individual and contains:
  - the original feature values & model prediction
  - all counterfactuals produced by DiCE (with per-CF confidence)
  - DiCE-paper quality metrics (proximity, sparsity, diversity …)
  - model-level performance context
  - a `ground_truth_issues` placeholder for Ivan's issue taxonomy

Usage
-----
From the repo root:

    $env:PYTHONPATH="src"; python src/case_builder.py          # writes results/cases.json
    $env:PYTHONPATH="src"; python src/case_builder.py --pretty  # human-readable indent

Programmatic:

    from case_builder import build_cases
    cases = build_cases()          # list[dict] ready for run_debate()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths (relative to repo root — run with PYTHONPATH=src from repo root)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
UNFAVORABLE_PATH = RESULTS_DIR / "unfavorable_samples.csv"
COUNTERFACTUALS_PATH = RESULTS_DIR / "counterfactuals.csv"
METRICS_PATH = RESULTS_DIR / "cf_metrics_per_instance.csv"
MODEL_METRICS_PATH = RESULTS_DIR / "logistic_regression_metrics.json"
OUTPUT_PATH = RESULTS_DIR / "cases.json"

# Features that the pipeline declares as immutable during CF generation
IMMUTABLE_FEATURES = [
    "age", "fnlwgt", "education", "education-num",
    "marital-status", "relationship", "race", "sex", "native-country",
]

# The 14 raw features in the Adult Income dataset
FEATURE_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]

LABEL_MAP = {0: "<=50K", 1: ">50K"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_python(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for JSON serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _row_to_features(row: pd.Series) -> dict[str, Any]:
    """Extract the 14 raw feature columns from a DataFrame row."""
    return {col: _safe_python(row[col]) for col in FEATURE_COLS}


def _compute_changes(original: dict, cf: dict) -> tuple[list[str], dict]:
    """Return (features_changed, changes_summary) comparing original vs CF."""
    changed: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    for feat in FEATURE_COLS:
        orig_val = original.get(feat)
        cf_val = cf.get(feat)
        if orig_val != cf_val:
            changed.append(feat)
            summary[feat] = {"from": orig_val, "to": cf_val}
    return changed, summary


def _load_model_metrics() -> dict[str, Any]:
    """Load model-level performance from the training step."""
    if not MODEL_METRICS_PATH.exists():
        return {}
    with open(MODEL_METRICS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # Keep only the scalar metrics the agents need
    return {
        "name": "logistic_regression",
        "accuracy": raw.get("accuracy"),
        "precision": raw.get("precision"),
        "recall": raw.get("recall"),
        "f1": raw.get("f1"),
    }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_cases(
    *,
    label_fn: Callable[[dict], list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Read all pipeline outputs and assemble one case dict per unfavorable
    individual.

    Parameters
    ----------
    label_fn : callable, optional
        A function ``(case_dict) -> list[str]`` that assigns ground-truth
        issue labels.  When Ivan's taxonomy is ready, plug it in here.
        Defaults to an empty list per case.

    Returns
    -------
    list[dict]
        Case dicts ready to be passed to ``debate.run_debate(case_data)``.
    """
    # --- Load pipeline artefacts ----------------------------------------
    samples = pd.read_csv(UNFAVORABLE_PATH)
    cf_all = pd.read_csv(COUNTERFACTUALS_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    model_info = _load_model_metrics()

    # Index metrics by original_index for quick lookup
    metrics_map: dict[int, dict] = {}
    for _, mrow in metrics.iterrows():
        idx = int(mrow["original_index"])
        metrics_map[idx] = {
            col: _safe_python(mrow[col])
            for col in metrics.columns
            if col != "original_index"
        }

    cases: list[dict[str, Any]] = []

    for i, srow in samples.iterrows():
        original_features = _row_to_features(srow)

        # Prediction context from predict.py
        prediction_label = LABEL_MAP.get(int(srow["prediction"]), str(srow["prediction"]))
        true_label = LABEL_MAP.get(int(srow["true_label"]), str(srow["true_label"]))

        # --- Gather counterfactuals for this instance -------------------
        instance_cfs = cf_all[
            (cf_all["original_index"] == i) & (cf_all["row_type"] == "counterfactual")
        ]

        counterfactuals: list[dict[str, Any]] = []
        for _, cf_row in instance_cfs.iterrows():
            cf_features = _row_to_features(cf_row)
            changed, summary = _compute_changes(original_features, cf_features)
            counterfactuals.append({
                "cf_rank": int(cf_row["cf_rank"]),
                "cf_confidence": _safe_python(cf_row["cf_confidence"]),
                "features": cf_features,
                "features_changed": changed,
                "changes_summary": summary,
            })

        # --- Assemble the case dict -------------------------------------
        case: dict[str, Any] = {
            "case_id": int(i),
            "domain": "income_prediction",
            "model_info": model_info,
            "original": original_features,
            "prediction": prediction_label,
            "prediction_confidence": _safe_python(1.0 - srow["proba"]),
            "true_label": true_label,
            "is_false_negative": bool(srow["is_false_negative"]),
            "counterfactuals": counterfactuals,
            "metrics": metrics_map.get(int(i), {}),
            "ground_truth_issues": [],  # ← Ivan's taxonomy plugs in here
        }

        # Optional: auto-label if a taxonomy function is provided
        if label_fn is not None:
            case["ground_truth_issues"] = label_fn(case)

        cases.append(case)

    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build debate cases from pipeline outputs."
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Write indented JSON for human inspection.",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Output path (default: {OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    cases = build_cases()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2 if args.pretty else None, ensure_ascii=False)

    print(f"Built {len(cases)} cases → {out}")
    for c in cases:
        n_cf = len(c["counterfactuals"])
        print(f"  case {c['case_id']:>2}: {n_cf} CFs, "
              f"prediction={c['prediction']}, "
              f"is_false_negative={c['is_false_negative']}")


if __name__ == "__main__":
    main()
