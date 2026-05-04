"""Issue taxonomy used by agents when evaluating counterfactual explanations."""

from __future__ import annotations


ISSUE_TAXONOMY: dict[str, str] = {
    # ---------------------------------------------------------------------------
    # Work Plausibility
    # ---------------------------------------------------------------------------
    "inconsistent_work_profile": (
        "The counterfactual proposes work-related edits that are internally "
        "inconsistent or temporally implausible, especially changes involving "
        "workclass and occupation."
    ),

    # ---------------------------------------------------------------------------
    # Working-Hours Realism
    # ---------------------------------------------------------------------------
    "extreme_working_hours": (
        "The counterfactual proposes an unrealistic or extreme change in "
        "hours_per_week, such as dangerously high hours, implausibly low hours, "
        "or a very large jump from the original value."
    ),

    # ---------------------------------------------------------------------------
    # Financial Actionability
    # ---------------------------------------------------------------------------
    "unactionable_capital_shift": (
        "The counterfactual relies on a capital_gain or capital_loss change "
        "that may be financially unrealistic or not realistically actionable "
        "for the individual."
    ),

    # ---------------------------------------------------------------------------
    # Burden and Actionability
    # ---------------------------------------------------------------------------
    "too_many_changes": (
        "The counterfactual modifies too many actionable features at once, "
        "placing an unrealistic burden on the individual."
    ),

    # ---------------------------------------------------------------------------
    # Prediction Robustness
    # ---------------------------------------------------------------------------
    "fragile_counterfactual": (
        "The counterfactual barely reaches the favorable class, with "
        "cf_confidence close to the decision threshold."
    ),
}

CONSTRAINT_VIOLATION_GUIDANCE: dict[str, str] = {
    "frozen_feature_changed": (
        "A frozen, immutable, protected, or non-actionable feature changed. "
        "This indicates a pipeline or preprocessing problem, not a scored "
        "counterfactual-quality issue."
    ),

    "invalid_value": (
        "A feature contains an invalid, missing, NaN, unknown, or semantically "
        "invalid value. This should be treated as a technical warning unless "
        "it directly affects the quality of the recommendation."
    ),

    "outside_permitted_range": (
        "A generated counterfactual value is outside the permitted DiCE range. "
        "This should be treated as a constraint or pipeline violation."
    ),
}

def get_issue_guidance() -> str:
    """Format the scored taxonomy as a bullet list for injection into prompts."""
    labels = "\n".join(
        f"- {label}: {description}"
        for label, description in ISSUE_TAXONOMY.items()
    )

    return (
        "Use only the following scored issue labels in final verdicts:\n"
        f"{labels}\n\n"
        "Do not invent new scored labels. Do not use constraint-violation labels "
        "as scored issues."
    )

def get_constraint_guidance() -> str:
    """Format constraint-violation guidance for prompt injection."""
    labels = "\n".join(
        f"- {label}: {description}"
        for label, description in CONSTRAINT_VIOLATION_GUIDANCE.items()
    )

    return (
        "Constraint violations are not scored issue labels. They indicate possible "
        "pipeline, preprocessing, or DiCE-generation problems.\n"
        f"{labels}\n\n"
        "The Judge may mention constraint violations in the rationale, but must "
        "not include them in `flagged_issues`."
    )

def get_evidence_guidance() -> str:
    """Explain how agents should use heuristic evidence."""
    return (
        "Use `heuristic_metrics.flagged_issues` as deterministic scored issue labels.\n"
        "Use `heuristic_metrics.issue_evidence` to explain why an issue was flagged. "
        "When evidence gives numeric values, thresholds, or deltas, rely on those "
        "values instead of recalculating or guessing.\n"
        "Use `heuristic_metrics.constraint_violations` and "
        "`heuristic_metrics.constraint_evidence` separately from scored issues. "
        "These fields indicate possible pipeline/debugging problems, not ordinary "
        "counterfactual-quality flaws.\n"
        "Use `cf_confidence` when discussing prediction robustness. A confidence "
        "close to 0.5 indicates a fragile counterfactual."
    )