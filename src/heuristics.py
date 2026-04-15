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
    
    Args:
        original_row: Original instance as a dict-like object.
                      IMPORTANT: Feature names must be underscored (education_num, not education-num).
        cf_row: Counterfactual instance as a dict-like object.
                IMPORTANT: Feature names must be underscored.

    Returns:
        dict with:
            - changed_features: list[str]
            - changes: dict[str, dict[str, object]] with old/new values
            - sparsity: int
            - age_delta: int | None
            - flagged_issues: list[str] using taxonomy keys
    """
    # DiCE-style pipelines output full rows. Compute the feature-level diff here.
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

            # Treat NaN->NaN as unchanged.
            if math.isnan(original_float) and math.isnan(new_float):
                return False

            return not math.isclose(original_float, new_float, rel_tol=rel_tol, abs_tol=abs_tol)

        return new_value != original_value

    changed_features = []
    changes = {}
    for feature, original_value in original_row.items():
        if feature not in cf_row:
            continue
        new_value = cf_row[feature]
        if values_differ(original_value, new_value):
            changed_features.append(feature)
            changes[feature] = {"old": original_value, "new": new_value}

    flagged_issues = []

    # Ordinal hierarchy for checking implausible education jumps.
    # Note: Using hyphenated values as seen in the DiCE counterfactuals.csv output.
    education_hierarchy = [
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad",
        "Some-college",
        "Assoc-voc",
        "Assoc-acdm",
        "Bachelors",
        "Masters",
        "Prof-school",
        "Doctorate"
    ]
    
    # Static mapping to ensure education strictly matches education-num.
    education_to_num = {
        "Preschool": 1,
        "1st-4th": 2,
        "5th-6th": 3,
        "7th-8th": 4,
        "9th": 5,
        "10th": 6,
        "11th": 7,
        "12th": 8,
        "HS-grad": 9,
        "Some-college": 10,
        "Assoc-voc": 11,
        "Assoc-acdm": 12,
        "Bachelors": 13,
        "Masters": 14,
        "Prof-school": 15,
        "Doctorate": 16
    }
    
    education_num_changed = "education_num" in changed_features
    if "education" in changed_features or education_num_changed:
        new_edu = cf_row.get("education")
        new_edu_num = cf_row.get("education_num")
        
        # 1. Check education_mismatch
        if new_edu is not None and new_edu_num is not None:
            expected_num = education_to_num.get(new_edu)
            try:
                if expected_num != int(new_edu_num):
                    flagged_issues.append("education_mismatch")
            except (TypeError, ValueError):
                flagged_issues.append("education_mismatch")
                
        # 2. Check implausible_education_jump
        if "education" in changed_features:
            old_edu = original_row.get("education")
            if old_edu in education_hierarchy and new_edu in education_hierarchy:
                old_idx = education_hierarchy.index(old_edu)
                new_idx = education_hierarchy.index(new_edu)
                # A jump of > 1 level is flagged as implausible
                if abs(new_idx - old_idx) > 1:
                    flagged_issues.append("implausible_education_jump")


    # Granular protected-attribute labels aligned with ISSUE_TAXONOMY.
    fairness_feature_to_issue = {
        "sex": "sex_change",
        "race": "race_change",
        "native_country": "native_country_change",
    }
    for feature, issue_label in fairness_feature_to_issue.items():
        if feature in changed_features:
            flagged_issues.append(issue_label)

    # Keep deterministic non-fairness checks that are simple arithmetic/string checks.
    if "fnlwgt" in changed_features:
        flagged_issues.append("fnlwgt_change")

    age_delta = None
    if "age" in changed_features:
        original_age = original_row.get("age")
        new_age = cf_row.get("age")
        if original_age is not None and new_age is not None:
            age_delta = new_age - original_age
            if age_delta < 0:
                flagged_issues.append("age_reversal")
            elif age_delta > 10:
                flagged_issues.append("extreme_age_increase")

    # too_many_changes logic
    major_features = {"education", "hours_per_week", "workclass", "occupation", "marital_status", "capital_gain", "capital_loss"}
    minor_features = {"relationship", "education_num"}
    
    changed_major = set(changed_features).intersection(major_features)
    changed_minor = set(changed_features).intersection(minor_features)
    
    major_count = len(changed_major)
    minor_count = len(changed_minor)
    
    # Coupled heuristics: count "workclass" and "occupation" changes as 1 major change if both change
    if "workclass" in changed_major and "occupation" in changed_major:
        major_count -= 1
        
    # Coupled heuristics: count "education" and "education_num" as 1 major change
    if "education" in changed_major and "education_num" in changed_minor:
        minor_count -= 1 # The major count already includes education
        
    if major_count >= 3 or (major_count == 2 and minor_count >= 2):
        flagged_issues.append("too_many_changes")

    metrics = {
        "changed_features": changed_features,
        "changes": changes,
        "sparsity": len(changed_features),
        "age_delta": age_delta,
        "flagged_issues": flagged_issues,
    }
                
    return metrics

# Quick testing example
if __name__ == "__main__":
    # Using Adult Income dataset features
    person = {
      "age": 25,
      "hours_per_week": 40,
      "workclass": "Private",
      "sex": "Female",
      "race": "White",
      "native_country": "United-States",
      "fnlwgt": 120000
    }
    
    # Full-row counterfactuals (as produced in CSV rows), not "changes-only" dicts.
    cf_1 = {
        "age": 25,
        "hours_per_week": 45,
        "workclass": "Self-emp-not-inc",
        "sex": "Female",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 120000,
    }
    cf_2 = {
        "age": 25,
        "hours_per_week": 40,
        "workclass": "Private",
        "sex": "Male",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 125000,
    }
    cf_3 = {
        "age": 18,
        "hours_per_week": 40,
        "workclass": "Private",
        "sex": "Female",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 120000,
    }  # Age reversal
    cf_4 = {
        "age": 45,
        "hours_per_week": 40,
        "workclass": "Private",
        "sex": "Female",
        "race": "White",
        "native_country": "United-States",
        "fnlwgt": 120000,
    }  # Extreme increase
    
    print("CF 1 (Work edits only):", compute_heuristic_metrics(person, cf_1))
    print("CF 2 (Sex + fnlwgt):", compute_heuristic_metrics(person, cf_2))
    print("CF 3 (Age reversal):", compute_heuristic_metrics(person, cf_3))
    print("CF 4 (Extreme age increase):", compute_heuristic_metrics(person, cf_4))