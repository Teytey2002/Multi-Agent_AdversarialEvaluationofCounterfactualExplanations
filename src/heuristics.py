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

def compute_heuristic_metrics(original_row, cf_row):
    """
    Compute deterministic taxonomy-aligned metrics from full rows.

    The scored taxonomy only includes:
        - extreme_working_hours
        - inconsistent_work_profile
        - unactionable_capital_shift
        - too_many_changes

    Frozen-feature changes are recorded separately as constraint violations.
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

    flagged_issues = []
    constraint_violations = []

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
    # 2. extreme_working_hours
    # ------------------------------------------------------------
    if "hours_per_week" in changed_features:
        new_hours = cf_row.get("hours_per_week")

        try:
            new_hours = float(new_hours)

            if new_hours < 20 or new_hours > 50:
                flagged_issues.append("extreme_working_hours")

        except (TypeError, ValueError):
            constraint_violations.append("hours_per_week_invalid_value")

    # ------------------------------------------------------------
    # 3. unactionable_capital_shift
    # ------------------------------------------------------------
    for capital_feature in ["capital_gain", "capital_loss"]:
        if capital_feature in changed_features:
            new_value = cf_row.get(capital_feature)

            try:
                new_value = float(new_value)

                if new_value < 0 or new_value > 5000:
                    flagged_issues.append("unactionable_capital_shift")

            except (TypeError, ValueError):
                constraint_violations.append(
                    f"{capital_feature}_invalid_value"
                )

    # ------------------------------------------------------------
    # 4. inconsistent_work_profile
    # ------------------------------------------------------------
    if "workclass" in changed_features and "occupation" not in changed_features:
        new_workclass = cf_row.get("workclass")
        old_occupation = original_row.get("occupation")

        if new_workclass in {"Without-pay", "Never-worked"} and old_occupation not in {None, "?"}:
            flagged_issues.append("inconsistent_work_profile")

    if "occupation" in changed_features and "workclass" not in changed_features:
        new_occupation = cf_row.get("occupation")
        old_workclass = original_row.get("workclass")

        if old_workclass in {"Without-pay", "Never-worked"} and new_occupation not in {None, "?"}:
            flagged_issues.append("inconsistent_work_profile")

    # ------------------------------------------------------------
    # 5. too_many_changes
    # ------------------------------------------------------------
    changed_actionable = set(changed_features).intersection(ACTIONABLE_FEATURES)

    burden_count = len(changed_actionable)

    # workclass + occupation may count as one logical intervention.
    if "workclass" in changed_actionable and "occupation" in changed_actionable:
        burden_count -= 1

    if burden_count >= 3:
        flagged_issues.append("too_many_changes")

    metrics = {
        "changed_features": changed_features,
        "changes": changes,
        "sparsity": len(changed_features),
        "flagged_issues": sorted(set(flagged_issues)),
        "constraint_violations": sorted(set(constraint_violations)),
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