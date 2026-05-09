"""
Deterministic bridge-layer heuristics for counterfactual evaluation.

The LLM agents should not be asked to count feature changes, compare numeric
thresholds, or infer policy violations from scratch. This module computes those
facts once, using the project feature policy, and injects them into each case.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import Any, Mapping

from policy.feature_policy import (
    ACTIONABLE_FEATURES_CANONICAL,
    AGE_MAX_INCREASE,
    CAPITAL_LARGE_JUMP_THRESHOLD,
    EDUCATION_NUM_MAX_INCREASE,
    FROZEN_FEATURES_CANONICAL,
    FRAGILITY_THRESHOLD,
    HOURS_MAX_DECREASE,
    HOURS_MAX_INCREASE,
    canonical_name,
    education_label_from_num,
    is_synchronized_education_label_change,
)


def _canonical_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw Adult feature names to taxonomy-style names."""
    return {canonical_name(str(key)): value for key, value in row.items()}


def _to_float(value: Any) -> float:
    """Convert a scalar value to float, preserving the caller's exception path."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric feature value")
    return float(value)


def _is_integer_like(value: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(value, round(value), abs_tol=tolerance)


def _normalise_permitted_range(
    permitted_range: Mapping[str, Any] | None,
) -> dict[str, tuple[float, float]]:
    """Normalize optional raw permitted_range keys to canonical feature names."""
    if not permitted_range:
        return {}

    ranges: dict[str, tuple[float, float]] = {}
    for feature, bounds in permitted_range.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        try:
            ranges[canonical_name(str(feature))] = (_to_float(bounds[0]), _to_float(bounds[1]))
        except (TypeError, ValueError):
            continue
    return ranges


def compute_heuristic_metrics(
    original_row: Mapping[str, Any],
    cf_row: Mapping[str, Any],
    cf_confidence: Any = None,
    permitted_range: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute deterministic taxonomy-aligned metrics from full rows.

    Scored issue labels:
        - extreme_working_hours
        - inconsistent_work_profile
        - implausible_time_dependent_change
        - unactionable_capital_shift
        - too_many_changes
        - fragile_counterfactual

    Constraint violations are recorded separately. They indicate a generation
    or policy-enforcement problem, not an ordinary scored issue.
    """

    original = _canonical_row(original_row)
    cf = _canonical_row(cf_row)
    permitted_ranges = _normalise_permitted_range(permitted_range)

    def values_differ(original_value: Any, new_value: Any, rel_tol=1e-6, abs_tol=1e-8) -> bool:
        """Robust equality check that is tolerant for numeric float artifacts."""
        if original_value is None and new_value is None:
            return False
        if original_value is None or new_value is None:
            return True

        is_original_numeric = isinstance(original_value, Number) and not isinstance(original_value, bool)
        is_new_numeric = isinstance(new_value, Number) and not isinstance(new_value, bool)

        if is_original_numeric and is_new_numeric:
            original_float = float(original_value)
            new_float = float(new_value)

            if math.isnan(original_float) and math.isnan(new_float):
                return False

            return not math.isclose(
                original_float,
                new_float,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )

        return new_value != original_value

    changed_features: list[str] = []
    changes: dict[str, dict[str, Any]] = {}

    for feature, original_value in original.items():
        if feature not in cf:
            continue

        new_value = cf[feature]

        if values_differ(original_value, new_value):
            if feature == "education" and is_synchronized_education_label_change(original, cf):
                continue
            changed_features.append(feature)
            changes[feature] = {
                "old": original_value,
                "new": new_value,
            }

    flagged_issues: set[str] = set()
    constraint_violations: set[str] = set()

    issue_evidence: dict[str, list[dict[str, Any]]] = {}
    constraint_evidence: dict[str, list[dict[str, Any]]] = {}

    def add_issue(label: str, evidence: dict[str, Any]) -> None:
        flagged_issues.add(label)
        issue_evidence.setdefault(label, []).append(evidence)

    def add_constraint_violation(label: str, evidence: dict[str, Any]) -> None:
        constraint_violations.add(label)
        constraint_evidence.setdefault(label, []).append(evidence)

    def check_permitted_range(feature: str, old_value: Any, new_value: Any) -> None:
        if feature not in permitted_ranges:
            return
        try:
            new_float = _to_float(new_value)
        except (TypeError, ValueError):
            add_constraint_violation(
                f"{feature}_invalid_value",
                {
                    "feature": feature,
                    "old": old_value,
                    "new": new_value,
                    "reason": f"{feature} could not be converted to float.",
                },
            )
            return

        permitted_min, permitted_max = permitted_ranges[feature]
        if new_float < permitted_min - 1e-6 or new_float > permitted_max + 1e-6:
            add_constraint_violation(
                f"{feature}_outside_permitted_range",
                {
                    "feature": feature,
                    "old": old_value,
                    "new": new_value,
                    "permitted_min": permitted_min,
                    "permitted_max": permitted_max,
                    "reason": (
                        f"{feature} is outside the per-instance DiCE permitted range."
                    ),
                },
            )

    # ------------------------------------------------------------
    # 1. Constraint violations: frozen features should not change.
    # ------------------------------------------------------------
    for feature in changed_features:
        if feature in FROZEN_FEATURES_CANONICAL:
            add_constraint_violation(
                f"{feature}_changed_despite_being_frozen",
                {
                    "feature": feature,
                    "old": original.get(feature),
                    "new": cf.get(feature),
                    "reason": (
                        "This feature is frozen during counterfactual generation; "
                        "a change indicates a pipeline or constraint violation."
                    ),
                },
            )

    if "education" in changed_features:
        expected_label = education_label_from_num(cf.get("education_num"))
        add_constraint_violation(
            "education_changed_without_education_num_sync",
            {
                "feature": "education",
                "old": original.get("education"),
                "new": cf.get("education"),
                "education_num": cf.get("education_num"),
                "expected_education": expected_label,
                "reason": (
                    "education is a derived display label and may only change "
                    "to match a changed education_num value."
                ),
            },
        )

    # ------------------------------------------------------------
    # 2. fragile_counterfactual
    # ------------------------------------------------------------
    if cf_confidence is not None:
        try:
            confidence = _to_float(cf_confidence)
            decision_threshold = 0.5

            if decision_threshold <= confidence < FRAGILITY_THRESHOLD:
                add_issue(
                    "fragile_counterfactual",
                    {
                        "cf_confidence": confidence,
                        "decision_threshold": decision_threshold,
                        "fragility_threshold": FRAGILITY_THRESHOLD,
                        "margin_above_threshold": confidence - decision_threshold,
                        "reason": (
                            "The counterfactual reaches the favorable class, "
                            "but only barely."
                        ),
                    },
                )

        except (TypeError, ValueError):
            add_constraint_violation(
                "cf_confidence_invalid_value",
                {
                    "value": cf_confidence,
                    "reason": "cf_confidence could not be converted to float.",
                },
            )

    # ------------------------------------------------------------
    # 3. implausible_time_dependent_change
    # ------------------------------------------------------------
    age_delta: float | None = None
    education_delta: float | None = None

    if "age" in changed_features:
        old_age = original.get("age")
        new_age = cf.get("age")
        check_permitted_range("age", old_age, new_age)
        try:
            old_age_float = _to_float(old_age)
            new_age_float = _to_float(new_age)
            age_delta = new_age_float - old_age_float

            if not _is_integer_like(new_age_float):
                add_issue(
                    "implausible_time_dependent_change",
                    {
                        "feature": "age",
                        "old": old_age_float,
                        "new": new_age_float,
                        "reason": "Adult age should remain an integer-valued feature.",
                    },
                )

            if age_delta < 0:
                add_issue(
                    "implausible_time_dependent_change",
                    {
                        "feature": "age",
                        "old": old_age_float,
                        "new": new_age_float,
                        "delta": age_delta,
                        "reason": "The recourse policy allows age to increase, not decrease.",
                    },
                )

            if age_delta > AGE_MAX_INCREASE:
                add_issue(
                    "implausible_time_dependent_change",
                    {
                        "feature": "age",
                        "old": old_age_float,
                        "new": new_age_float,
                        "delta": age_delta,
                        "max_allowed_increase": AGE_MAX_INCREASE,
                        "reason": "The proposed age jump is too large for the configured horizon.",
                    },
                )

        except (TypeError, ValueError):
            add_constraint_violation(
                "age_invalid_value",
                {
                    "old": old_age,
                    "new": new_age,
                    "reason": "age could not be converted to float.",
                },
            )
    else:
        try:
            age_delta = 0.0
        except (TypeError, ValueError):
            age_delta = None

    if "education_num" in changed_features:
        old_education = original.get("education_num")
        new_education = cf.get("education_num")
        check_permitted_range("education_num", old_education, new_education)
        try:
            old_education_float = _to_float(old_education)
            new_education_float = _to_float(new_education)
            education_delta = new_education_float - old_education_float

            if not _is_integer_like(new_education_float):
                add_issue(
                    "implausible_time_dependent_change",
                    {
                        "feature": "education_num",
                        "old": old_education_float,
                        "new": new_education_float,
                        "reason": "education_num should correspond to an integer education level.",
                    },
                )

            if education_delta < 0:
                add_issue(
                    "implausible_time_dependent_change",
                    {
                        "feature": "education_num",
                        "old": old_education_float,
                        "new": new_education_float,
                        "delta": education_delta,
                        "reason": "The recourse policy allows education_num to increase, not decrease.",
                    },
                )

            if education_delta > EDUCATION_NUM_MAX_INCREASE:
                add_issue(
                    "implausible_time_dependent_change",
                    {
                        "feature": "education_num",
                        "old": old_education_float,
                        "new": new_education_float,
                        "delta": education_delta,
                        "max_allowed_increase": EDUCATION_NUM_MAX_INCREASE,
                        "reason": "The proposed education jump is too large for the configured horizon.",
                    },
                )

            if education_delta > 0:
                effective_age_delta = age_delta
                if effective_age_delta is None:
                    try:
                        effective_age_delta = _to_float(cf.get("age")) - _to_float(original.get("age"))
                    except (TypeError, ValueError):
                        effective_age_delta = None

                if effective_age_delta is None or effective_age_delta <= 0:
                    add_issue(
                        "implausible_time_dependent_change",
                        {
                            "feature": "education_num",
                            "old": old_education_float,
                            "new": new_education_float,
                            "delta": education_delta,
                            "age_delta": effective_age_delta,
                            "reason": (
                                "A higher education level requires time to pass, "
                                "but age did not increase."
                            ),
                        },
                    )
                elif effective_age_delta < education_delta:
                    add_issue(
                        "implausible_time_dependent_change",
                        {
                            "feature": "education_num",
                            "old": old_education_float,
                            "new": new_education_float,
                            "delta": education_delta,
                            "age_delta": effective_age_delta,
                            "min_age_years_per_education_num_level": 1,
                            "reason": (
                                "The education increase is too large for the age increase."
                            ),
                        },
                    )

        except (TypeError, ValueError):
            add_constraint_violation(
                "education_num_invalid_value",
                {
                    "old": old_education,
                    "new": new_education,
                    "reason": "education_num could not be converted to float.",
                },
            )

    # ------------------------------------------------------------
    # 4. extreme_working_hours
    # ------------------------------------------------------------
    if "hours_per_week" in changed_features:
        old_hours = original.get("hours_per_week")
        new_hours = cf.get("hours_per_week")
        check_permitted_range("hours_per_week", old_hours, new_hours)

        try:
            old_hours_float = _to_float(old_hours)
            new_hours_float = _to_float(new_hours)
            hours_delta = new_hours_float - old_hours_float

            if (
                hours_delta <= -HOURS_MAX_DECREASE
                or hours_delta >= HOURS_MAX_INCREASE
                or new_hours_float <= 20
                or new_hours_float >= 60
            ):
                add_issue(
                    "extreme_working_hours",
                    {
                        "old": old_hours_float,
                        "new": new_hours_float,
                        "delta": hours_delta,
                        "large_decrease_threshold": HOURS_MAX_DECREASE,
                        "large_increase_threshold": HOURS_MAX_INCREASE,
                        "low_hours_extreme": 20,
                        "high_hours_extreme": 60,
                        "reason": (
                            "The counterfactual proposes an extreme or large "
                            "change in weekly working hours."
                        ),
                    },
                )

        except (TypeError, ValueError):
            add_constraint_violation(
                "hours_per_week_invalid_value",
                {
                    "old": old_hours,
                    "new": new_hours,
                    "reason": "hours_per_week could not be converted to float.",
                },
            )

    # ------------------------------------------------------------
    # 5. unactionable_capital_shift
    # ------------------------------------------------------------
    for capital_feature in ["capital_gain", "capital_loss"]:
        if capital_feature in changed_features:
            old_value = original.get(capital_feature)
            new_value = cf.get(capital_feature)
            check_permitted_range(capital_feature, old_value, new_value)

            try:
                old_value_float = _to_float(old_value)
                new_value_float = _to_float(new_value)
                delta = new_value_float - old_value_float

                if delta >= CAPITAL_LARGE_JUMP_THRESHOLD or (
                    old_value_float == 0 and new_value_float >= CAPITAL_LARGE_JUMP_THRESHOLD
                ):
                    add_issue(
                        "unactionable_capital_shift",
                        {
                            "feature": capital_feature,
                            "old": old_value_float,
                            "new": new_value_float,
                            "delta": delta,
                            "large_jump_threshold": CAPITAL_LARGE_JUMP_THRESHOLD,
                            "reason": (
                                "The counterfactual relies on a large financial "
                                "shift that may not be realistically actionable."
                            ),
                        },
                    )

            except (TypeError, ValueError):
                add_constraint_violation(
                    f"{capital_feature}_invalid_value",
                    {
                        "feature": capital_feature,
                        "old": old_value,
                        "new": new_value,
                        "reason": f"{capital_feature} could not be converted to float.",
                    },
                )

    # ------------------------------------------------------------
    # 6. inconsistent_work_profile
    # ------------------------------------------------------------
    if "workclass" in changed_features and "occupation" not in changed_features:
        new_workclass = cf.get("workclass")
        old_occupation = original.get("occupation")

        if new_workclass in {"Without-pay", "Never-worked"} and old_occupation not in {None, "?"}:
            add_issue(
                "inconsistent_work_profile",
                {
                    "old_workclass": original.get("workclass"),
                    "new_workclass": cf.get("workclass"),
                    "old_occupation": original.get("occupation"),
                    "new_occupation": cf.get("occupation"),
                    "reason": (
                        "The workclass/occupation combination appears internally "
                        "inconsistent or temporally implausible."
                    ),
                },
            )

    if "occupation" in changed_features and "workclass" not in changed_features:
        new_occupation = cf.get("occupation")
        old_workclass = original.get("workclass")

        if old_workclass in {"Without-pay", "Never-worked"} and new_occupation not in {None, "?"}:
            add_issue(
                "inconsistent_work_profile",
                {
                    "old_workclass": original.get("workclass"),
                    "new_workclass": cf.get("workclass"),
                    "old_occupation": original.get("occupation"),
                    "new_occupation": cf.get("occupation"),
                    "reason": (
                        "The workclass/occupation combination appears internally "
                        "inconsistent or temporally implausible."
                    ),
                },
            )

    # ------------------------------------------------------------
    # 7. too_many_changes
    # ------------------------------------------------------------
    changed_actionable = set(changed_features).intersection(ACTIONABLE_FEATURES_CANONICAL)
    burden_count = len(changed_actionable)

    coupled_workclass_occupation = (
        "workclass" in changed_actionable and "occupation" in changed_actionable
    )
    coupled_age_education = (
        "age" in changed_actionable and "education_num" in changed_actionable
    )

    # Coupled features may count as one logical intervention.
    if coupled_workclass_occupation:
        burden_count -= 1
    if coupled_age_education:
        burden_count -= 1

    if burden_count >= 3:
        add_issue(
            "too_many_changes",
            {
                "changed_actionable_features": sorted(changed_actionable),
                "burden_count": burden_count,
                "raw_changed_feature_count": len(changed_features),
                "coupled_workclass_occupation": coupled_workclass_occupation,
                "coupled_age_education": coupled_age_education,
                "reason": (
                    "The counterfactual modifies too many actionable features "
                    "at once."
                ),
            },
        )

    return {
        "changed_features": changed_features,
        "changes": changes,
        "sparsity": len(changed_features),
        "actionable_sparsity": len(changed_actionable),
        "burden_count": burden_count,
        "flagged_issues": sorted(flagged_issues),
        "constraint_violations": sorted(constraint_violations),
        "issue_evidence": issue_evidence,
        "constraint_evidence": constraint_evidence,
    }


if __name__ == "__main__":
    person = {
        "age": 25,
        "education_num": 9,
        "hours_per_week": 40,
        "workclass": "Private",
        "occupation": "Sales",
        "sex": "Female",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 120000,
        "capital_gain": 0,
        "capital_loss": 0,
    }

    cf = {
        **person,
        "age": 26,
        "education_num": 12,
        "capital_gain": 6000,
    }

    print(compute_heuristic_metrics(person, cf, cf_confidence=0.55))
