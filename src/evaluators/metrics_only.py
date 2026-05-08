"""Metrics-only baseline for counterfactual evaluation.

This module converts the already-computed pipeline evidence into the same
verdict schema used by the LLM Judge. It is intentionally deterministic: it
does not replace human reference labels, but gives the project a non-LLM
baseline to compare against single-LLM and multi-agent outputs.
"""

from __future__ import annotations

from typing import Any

from agents.prompts import get_valid_issue_labels


CRITICAL_ISSUES = {
    "too_many_changes",
    "unactionable_capital_shift",
    "implausible_time_dependent_change",
    "extreme_working_hours",
    "inconsistent_work_profile",
}

METRIC_WARNING_THRESHOLDS = {
    "validity_min": 1.0,
    "sparsity_low": 0.65,
    "continuous_proximity_low": -1.0,
    "categorical_proximity_low": 0.70,
}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_issue_union(case: dict[str, Any]) -> list[str]:
    """Collect deterministic issue labels from case-level or CF-level evidence."""
    valid_labels = get_valid_issue_labels()
    issue_set = {
        str(label)
        for label in case.get("heuristic_summary", {}).get("flagged_issues_union", [])
        if str(label) in valid_labels
    }

    if not issue_set:
        for cf in case.get("counterfactuals", []):
            for label in cf.get("heuristic_metrics", {}).get("flagged_issues", []):
                if str(label) in valid_labels:
                    issue_set.add(str(label))

    return sorted(issue_set)


def _collect_constraint_union(case: dict[str, Any]) -> list[str]:
    """Collect deterministic constraint violations from case evidence."""
    constraint_set = {
        str(label)
        for label in case.get("heuristic_summary", {}).get("constraint_violations_union", [])
    }

    if not constraint_set:
        for cf in case.get("counterfactuals", []):
            for label in cf.get("heuristic_metrics", {}).get("constraint_violations", []):
                constraint_set.add(str(label))

    return sorted(constraint_set)


def _metric_warnings(metrics: dict[str, Any]) -> list[str]:
    """Return non-taxonomy metric warnings used for severity and rationale."""
    warnings: list[str] = []

    validity = _as_float(metrics.get("validity"))
    if validity is not None and validity < METRIC_WARNING_THRESHOLDS["validity_min"]:
        warnings.append("validity_below_one")

    sparsity = _as_float(metrics.get("sparsity"))
    if sparsity is not None and sparsity < METRIC_WARNING_THRESHOLDS["sparsity_low"]:
        warnings.append("low_sparsity")

    continuous_proximity = _as_float(metrics.get("continuous_proximity"))
    if (
        continuous_proximity is not None
        and continuous_proximity < METRIC_WARNING_THRESHOLDS["continuous_proximity_low"]
    ):
        warnings.append("low_continuous_proximity")

    categorical_proximity = _as_float(metrics.get("categorical_proximity"))
    if (
        categorical_proximity is not None
        and categorical_proximity < METRIC_WARNING_THRESHOLDS["categorical_proximity_low"]
    ):
        warnings.append("low_categorical_proximity")

    return warnings


def _derive_severity(
    issues: list[str],
    constraints: list[str],
    metric_warnings: list[str],
    is_false_negative: bool,
) -> str:
    if constraints:
        return "high"

    critical_count = len(set(issues).intersection(CRITICAL_ISSUES))
    if critical_count >= 2:
        return "high"
    if critical_count == 1:
        return "medium"
    if issues:
        return "low"
    if metric_warnings or is_false_negative:
        return "low"
    return "low"


def _derive_assessment(
    issues: list[str],
    constraints: list[str],
    severity: str,
    metric_warnings: list[str],
    is_false_negative: bool,
) -> str:
    if constraints or severity == "high":
        return "unfair"
    if issues or metric_warnings or is_false_negative:
        return "ambiguous"
    return "fair"


def _recommended_action(overall_assessment: str, severity: str) -> str:
    if overall_assessment == "fair":
        return "accept"
    if overall_assessment == "unfair" or severity == "high":
        return "reject"
    return "review"


def _reasoning_summary(
    issues: list[str],
    constraints: list[str],
    metric_warnings: list[str],
    is_false_negative: bool,
) -> str:
    parts: list[str] = []
    if issues:
        parts.append("deterministic heuristics flagged " + ", ".join(issues))
    if constraints:
        parts.append("constraint violations were detected")
    if metric_warnings:
        parts.append("metric warnings: " + ", ".join(metric_warnings))
    if is_false_negative:
        parts.append("the original prediction is a false negative")
    if not parts:
        return "No deterministic issue labels or metric warnings were triggered."

    summary = "; ".join(parts) + "."
    if len(summary.split()) > 60:
        return "Deterministic heuristics and metrics indicate review is needed."
    return summary


def evaluate_case_metrics_only(case: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic verdict for one case."""
    issues = _collect_issue_union(case)
    constraints = _collect_constraint_union(case)
    metric_warnings = _metric_warnings(case.get("metrics", {}))
    is_false_negative = bool(case.get("is_false_negative", False))

    severity = _derive_severity(issues, constraints, metric_warnings, is_false_negative)
    overall = _derive_assessment(
        issues,
        constraints,
        severity,
        metric_warnings,
        is_false_negative,
    )
    confidence = 0.95 if constraints else 0.85 if issues else 0.75

    verdict = {
        "case_id": int(case.get("case_id", -1)),
        "overall_assessment": overall,
        "flagged_issues": issues,
        "severity": severity,
        "confidence": confidence,
        "reasoning_summary": _reasoning_summary(
            issues,
            constraints,
            metric_warnings,
            is_false_negative,
        ),
        "recommended_action": _recommended_action(overall, severity),
    }

    if constraints:
        verdict["constraint_violations"] = constraints
    if metric_warnings:
        verdict["metric_warnings"] = metric_warnings

    return verdict


def evaluate_cases_metrics_only(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic verdicts for a list of cases."""
    return [evaluate_case_metrics_only(case) for case in cases]
