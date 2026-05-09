"""
Bridge layer — converts Theo's ML pipeline outputs into structured
case dicts that the AutoGen multi-agent debate can consume directly.

Each case corresponds to one unfavorable individual and contains:
  - the original feature values & model prediction
  - all counterfactuals produced by DiCE (with per-CF confidence)
  - DiCE-paper quality metrics (proximity, sparsity, diversity …)
  - model-level performance context
  - deterministic heuristic flags from policy.heuristics
  - draft reference labels from annotations/ground_truth_labels.json

Usage
-----
From the repo root:

    python -m pipeline.case_builder          # writes results/cases.json
    python -m pipeline.case_builder --pretty  # human-readable indent

Programmatic:

    from pipeline.case_builder import build_cases
    cases = build_cases()          # list[dict] ready for run_debate()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from policy.feature_policy import (
    FEATURE_ALIASES,
    RAW_FEATURE_COLUMNS,
    is_synchronized_education_label_change,
)
from policy.heuristics import compute_heuristic_metrics


# ---------------------------------------------------------------------------
# Paths (relative to repo root — run with PYTHONPATH=src from repo root)
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
ANNOTATIONS_DIR = Path("annotations")
UNFAVORABLE_PATH = RESULTS_DIR / "unfavorable_samples.csv"
COUNTERFACTUALS_PATH = RESULTS_DIR / "counterfactuals.csv"
METRICS_PATH = RESULTS_DIR / "cf_metrics_per_instance.csv"
MODEL_METRICS_PATH = RESULTS_DIR / "logistic_regression_metrics.json"
GENERATION_POLICY_PATH = RESULTS_DIR / "generation_policy.json"
GROUND_TRUTH_LABELS_PATH = ANNOTATIONS_DIR / "ground_truth_labels.json"
OUTPUT_PATH = RESULTS_DIR / "cases.json"

# The 14 raw features in the Adult Income dataset
FEATURE_COLS = list(RAW_FEATURE_COLUMNS)

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
    """Return (features_changed, changes_summary) comparing original vs CF.
    
    Uses underscored feature names for consistency with heuristics logic.
    """
    changed: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    for feat in FEATURE_COLS:
        orig_val = original.get(feat)
        cf_val = cf.get(feat)
        if orig_val != cf_val:
            if feat == "education" and is_synchronized_education_label_change(original, cf):
                continue
            # Convert hyphenated feature name to underscored for consistency
            canonical_feat = FEATURE_ALIASES.get(feat, feat)
            changed.append(canonical_feat)
            summary[canonical_feat] = {"from": orig_val, "to": cf_val}
    return changed, summary


def _with_aliases(row_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a heuristics-friendly row with canonical underscore feature names.
    
    Converts hyphenated feature names (from CSV) to underscored names for consistency
    with policy.heuristics logic and issue taxonomy.
    """
    out: dict[str, Any] = {}
    for key, value in row_dict.items():
        # Convert hyphenated names to underscored; pass through others unchanged
        canonical_key = FEATURE_ALIASES.get(key, key)
        out[canonical_key] = value

    return out


def _load_heuristic_fn() -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Return the compute_heuristic_metrics function (always available since imported)."""
    return compute_heuristic_metrics


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


def _load_generation_policy() -> dict[str, Any]:
    """Load DiCE generation-policy metadata when available."""
    if not GENERATION_POLICY_PATH.exists():
        return {}
    with open(GENERATION_POLICY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_ground_truth_annotations() -> dict[str, Any]:
    """Load manual/team ground-truth labels when available."""
    if not GROUND_TRUTH_LABELS_PATH.exists():
        return {}
    with open(GROUND_TRUTH_LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _policy_context_for_case(policy: dict[str, Any], case_id: int) -> dict[str, Any]:
    """Return compact policy metadata relevant to one case."""
    if not policy:
        return {}

    per_instance_range = policy.get("per_instance_permitted_range", {})
    context = {
        key: value
        for key, value in policy.items()
        if key != "per_instance_permitted_range"
    }
    context["permitted_range"] = per_instance_range.get(str(case_id), {})
    return context


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
        Defaults to labels loaded from annotations/ground_truth_labels.json,
        or an empty list when no annotation exists for a case.

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
    generation_policy = _load_generation_policy()
    ground_truth_annotations = _load_ground_truth_annotations()
    ground_truth_cases = ground_truth_annotations.get("cases", {})
    heuristic_fn = _load_heuristic_fn()
    per_instance_ranges = generation_policy.get("per_instance_permitted_range", {})

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
        case_policy = _policy_context_for_case(generation_policy, int(i))
        permitted_range = per_instance_ranges.get(str(i), {})
        ground_truth_case = ground_truth_cases.get(str(i), {})

        # Prediction context from predict.py
        prediction_label = LABEL_MAP.get(int(srow["prediction"]), str(srow["prediction"]))
        true_label = LABEL_MAP.get(int(srow["true_label"]), str(srow["true_label"]))

        # --- Gather counterfactuals for this instance -------------------
        instance_cfs = cf_all[
            (cf_all["original_index"] == i) & (cf_all["row_type"] == "counterfactual")
        ]

        counterfactuals: list[dict[str, Any]] = []
        aggregate_issue_labels: set[str] = set()
        aggregate_constraint_violations: set[str] = set()
        aggregate_issue_evidence: dict[str, list[dict[str, Any]]] = {}
        aggregate_constraint_evidence: dict[str, list[dict[str, Any]]] = {}
        for _, cf_row in instance_cfs.iterrows():
            cf_features = _row_to_features(cf_row)
            changed, summary = _compute_changes(original_features, cf_features)
            cf_confidence = _safe_python(cf_row["cf_confidence"])
            heuristic_metrics = heuristic_fn(
                _with_aliases(original_features),
                _with_aliases(cf_features),
                cf_confidence=cf_confidence,
                permitted_range=permitted_range,
            )
            for issue in heuristic_metrics.get("flagged_issues", []):
                aggregate_issue_labels.add(str(issue))

            for violation in heuristic_metrics.get("constraint_violations", []):
                aggregate_constraint_violations.add(str(violation))

            for label, evidence_items in heuristic_metrics.get("issue_evidence", {}).items():
                aggregate_issue_evidence.setdefault(label, []).extend(evidence_items)

            for label, evidence_items in heuristic_metrics.get("constraint_evidence", {}).items():
                aggregate_constraint_evidence.setdefault(label, []).extend(evidence_items)

            counterfactuals.append({
                "cf_rank": int(cf_row["cf_rank"]),
                "cf_confidence": cf_confidence,
                "features": cf_features,
                "features_changed": changed,
                "changes_summary": summary,
                "heuristic_metrics": heuristic_metrics,
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
            "generation_policy": case_policy,
            "heuristic_summary": {
                "flagged_issues_union": sorted(aggregate_issue_labels),
                "constraint_violations_union": sorted(aggregate_constraint_violations),
                "issue_evidence": aggregate_issue_evidence,
                "constraint_evidence": aggregate_constraint_evidence,
            },
            "ground_truth_issues": list(ground_truth_case.get("ground_truth_issues", [])),
            "ground_truth_by_cf": ground_truth_case.get("ground_truth_by_cf", {}),
            "ground_truth_source": {
                "file": GROUND_TRUTH_LABELS_PATH.as_posix(),
                "schema_version": ground_truth_annotations.get("schema_version"),
                "annotation_status": ground_truth_annotations.get("annotation_status"),
            } if ground_truth_case else None,
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

    print(f"Built {len(cases)} cases -> {out}")
    for c in cases:
        n_cf = len(c["counterfactuals"])
        print(f"  case {c['case_id']:>2}: {n_cf} CFs, "
              f"prediction={c['prediction']}, "
              f"is_false_negative={c['is_false_negative']}")


if __name__ == "__main__":
    main()
