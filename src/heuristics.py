"""
This file demonstrates the 'Bridge Layer' heuristics pattern.

In the AutoGen-based pipeline, we don't rely 
on LLM agents to accurately count features or perform math to calculate differences 
(tasks LLMs often struggle with). Instead, we will use this 'heuristic' concept:

1. Pre-computation: Python code will load the real outputs from DiCE (counterfactuals.csv) 
   and programmatically check them against the rules defined in our taxonomy.md.
2. Injection: These deterministic flags (e.g., 'Sparsity: 4', 'Age changed by: -5 years') 
   will be injected directly into the structured case prompt.
3. Agent Debate: The 'Expert Witness' agent will use these pre-computed facts as hard 
   evidence during the debate, ensuring the Judge's final verdict is grounded in 
   quantitative reality rather than AI hallucinations.
"""

import math
from numbers import Number

def compute_heuristic_metrics(original_row, cf_row, cf_confidence=None):
    """
    Compute deterministic taxonomy-aligned metrics from full rows.

    The scored taxonomy includes:
        - extreme_working_hours
        - inconsistent_work_profile
        - unactionable_capital_shift
        - too_many_changes
        - fragile_counterfactual

    Frozen-feature changes are recorded separately as constraint violations.

    Args:
        original_row: Original instance as a dict-like object.
        cf_row: Counterfactual instance as a dict-like object.
        cf_confidence: Optional model probability for the favorable class (>50K).
    """

    def values_differ(original_value, new_value, rel_tol=1e-6, abs_tol=1e-8):
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
                abs_tol=abs_tol
            )

        return new_value != original_value

    changed_features = []
    changes = {}

    for feature, original_value in original_row.items():
        if feature not in cf_row:
            continue

        new_value = cf_row[feature]

        if values_differ(original_value, new_value):
            changed_features.append(feature)
            changes[feature] = {
                "old": original_value,
                "new": new_value,
            }

    flagged_issues = set()
    constraint_violations = set()

    issue_evidence = {}
    constraint_evidence = {}

    def add_issue(label, evidence):
        flagged_issues.add(label)
        issue_evidence.setdefault(label, []).append(evidence)


    def add_constraint_violation(label, evidence):
        constraint_violations.add(label)
        constraint_evidence.setdefault(label, []).append(evidence)

    ACTIONABLE_FEATURES = {
        "workclass",
        "occupation",
        "hours_per_week",
        "capital_gain",
        "capital_loss",
    }

    FROZEN_FEATURES = {
        "age",
        "race",
        "sex",
        "native_country",
        "education",
        "marital_status",
        "relationship",
        "fnlwgt",
        "education_num",
    }

    # ------------------------------------------------------------
    # 1. Constraint violations: frozen features should not change.
    # ------------------------------------------------------------
    for feature in changed_features:
        if feature in FROZEN_FEATURES:
            constraint_violations.append(
                f"{feature}_changed_despite_being_frozen"
            )

    # ------------------------------------------------------------
    # 2. fragile_counterfactual
    # ------------------------------------------------------------
    if cf_confidence is not None:
        try:
            confidence = float(cf_confidence)
            decision_threshold = 0.5
            fragility_threshold = 0.60

            if decision_threshold <= confidence < fragility_threshold:
                add_issue(
                    "fragile_counterfactual",
                    {
                        "cf_confidence": confidence,
                        "decision_threshold": decision_threshold,
                        "fragility_threshold": fragility_threshold,
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
    # 3. extreme_working_hours
    # ------------------------------------------------------------
    if "hours_per_week" in changed_features:
        old_hours = original_row.get("hours_per_week")
        new_hours = cf_row.get("hours_per_week")

        try:
            old_hours = float(old_hours)
            new_hours = float(new_hours)
            hours_delta = new_hours - old_hours

            permitted_min = 20
            permitted_max = 50
            large_jump_threshold = 15

            # Pipeline sanity check: DiCE should not exceed permitted range.
            if new_hours < permitted_min or new_hours > permitted_max:
                add_constraint_violation(
                    "hours_per_week_outside_permitted_range",
                    {
                        "old": old_hours,
                        "new": new_hours,
                        "permitted_min": permitted_min,
                        "permitted_max": permitted_max,
                        "reason": (
                            "hours_per_week is outside the permitted DiCE range."
                        ),
                    },
                )

            # Scored issue: even inside the range, the recommendation may be extreme.
            if (
                new_hours <= permitted_min
                or new_hours >= permitted_max
                or abs(hours_delta) >= large_jump_threshold
            ):
                add_issue(
                    "extreme_working_hours",
                    {
                        "old": old_hours,
                        "new": new_hours,
                        "delta": hours_delta,
                        "large_jump_threshold": large_jump_threshold,
                        "permitted_min": permitted_min,
                        "permitted_max": permitted_max,
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
    # 4. unactionable_capital_shift
    # ------------------------------------------------------------
    for capital_feature in ["capital_gain", "capital_loss"]:
        if capital_feature in changed_features:
            old_value = original_row.get(capital_feature)
            new_value = cf_row.get(capital_feature)

            try:
                old_value = float(old_value)
                new_value = float(new_value)
                delta = new_value - old_value

                permitted_min = 0
                permitted_max = 5000
                large_jump_threshold = 3000

                # Pipeline sanity check: DiCE should not exceed permitted range.
                if new_value < permitted_min or new_value > permitted_max:
                    add_constraint_violation(
                        f"{capital_feature}_outside_permitted_range",
                        {
                            "feature": capital_feature,
                            "old": old_value,
                            "new": new_value,
                            "permitted_min": permitted_min,
                            "permitted_max": permitted_max,
                            "reason": (
                                f"{capital_feature} is outside the permitted DiCE range."
                            ),
                        },
                    )

                # Scored issue: large financial jumps can be unrealistic even inside range.
                if delta >= large_jump_threshold or (old_value == 0 and new_value >= large_jump_threshold):
                    add_issue(
                        "unactionable_capital_shift",
                        {
                            "feature": capital_feature,
                            "old": old_value,
                            "new": new_value,
                            "delta": delta,
                            "large_jump_threshold": large_jump_threshold,
                            "permitted_min": permitted_min,
                            "permitted_max": permitted_max,
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
    # 5. inconsistent_work_profile
    # ------------------------------------------------------------
    if "workclass" in changed_features and "occupation" not in changed_features:
        new_workclass = cf_row.get("workclass")
        old_occupation = original_row.get("occupation")

        if new_workclass in {"Without-pay", "Never-worked"} and old_occupation not in {None, "?"}:
            add_issue(
                "inconsistent_work_profile",
                {
                    "old_workclass": original_row.get("workclass"),
                    "new_workclass": cf_row.get("workclass"),
                    "old_occupation": original_row.get("occupation"),
                    "new_occupation": cf_row.get("occupation"),
                    "reason": (
                        "The workclass/occupation combination appears internally "
                        "inconsistent or temporally implausible."
                    ),
                },
            )

    if "occupation" in changed_features and "workclass" not in changed_features:
        new_occupation = cf_row.get("occupation")
        old_workclass = original_row.get("workclass")

        if old_workclass in {"Without-pay", "Never-worked"} and new_occupation not in {None, "?"}:
            add_issue(
                "inconsistent_work_profile",
                {
                    "old_workclass": original_row.get("workclass"),
                    "new_workclass": cf_row.get("workclass"),
                    "old_occupation": original_row.get("occupation"),
                    "new_occupation": cf_row.get("occupation"),
                    "reason": (
                        "The workclass/occupation combination appears internally "
                        "inconsistent or temporally implausible."
                    ),
                },
            )

    # ------------------------------------------------------------
    # 6. too_many_changes
    # ------------------------------------------------------------
    changed_actionable = set(changed_features).intersection(ACTIONABLE_FEATURES)

    burden_count = len(changed_actionable)

    # workclass + occupation may count as one logical intervention.
    if "workclass" in changed_actionable and "occupation" in changed_actionable:
        burden_count -= 1

    if burden_count >= 3:
        add_issue(
            "too_many_changes",
            {
                "changed_actionable_features": sorted(changed_actionable),
                "burden_count": burden_count,
                "raw_changed_feature_count": len(changed_features),
                "coupled_workclass_occupation": (
                    "workclass" in changed_actionable
                    and "occupation" in changed_actionable
                ),
                "reason": (
                    "The counterfactual modifies too many actionable features "
                    "at once."
                ),
            },
        )

    metrics = {
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

    return metrics

# Quick testing example
if __name__ == "__main__":
    person = {
        "age": 25,
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

    cf_1 = {
        "age": 25,
        "hours_per_week": 55,
        "workclass": "Private",
        "occupation": "Sales",
        "sex": "Female",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 120000,
        "capital_gain": 0,
        "capital_loss": 0,
    }

    cf_2 = {
        "age": 25,
        "hours_per_week": 45,
        "workclass": "Self-emp-not-inc",
        "occupation": "Exec-managerial",
        "sex": "Female",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 120000,
        "capital_gain": 6000,
        "capital_loss": 0,
    }

    cf_3 = {
        "age": 30,
        "hours_per_week": 40,
        "workclass": "Private",
        "occupation": "Sales",
        "sex": "Male",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 125000,
        "capital_gain": 0,
        "capital_loss": 0,
    }

    print("CF 1:", compute_heuristic_metrics(person, cf_1))
    print("CF 2:", compute_heuristic_metrics(person, cf_2))
    print("CF 3:", compute_heuristic_metrics(person, cf_3))