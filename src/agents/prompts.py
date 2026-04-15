"""Issue taxonomy used by agents when evaluating counterfactual explanations."""

from __future__ import annotations


ISSUE_TAXONOMY: dict[str, str] = {
    # ---------------------------------------------------------------------------
    # Fairness-Sensitive Issues (Most Severe)
    # ---------------------------------------------------------------------------
    "sex_change": "The CF modifies the individual's sex or gender.",
    "race_change": "The CF modifies the individual's race or ethnicity.",
    "native_country_change": "The CF alters the individual's native country.",
    
    # ---------------------------------------------------------------------------
    # Realism and Consistency
    # ---------------------------------------------------------------------------
    "age_reversal": "The CF decreases age, which is biologically impossible.",
    "implausible_education_jump": "Education jumps by multiple ordinal levels or to an implausible degree.",
    "extreme_working_hours": "The CF suggests dangerously high or implausibly low working hours.",
    "inconsistent_work_profile": "Work-related edits are internally inconsistent or temporally implausible.",
    "education_mismatch": "The education and education_num features become inconsistent with each other.",
    
    # ---------------------------------------------------------------------------
    # Burden and Actionability (Least Severe)
    # ---------------------------------------------------------------------------
    "extreme_age_increase": "The CF increases age by an extreme gap, rendering it unactionable.",
    "unactionable_capital_shift": "The CF suggests a financially unrealistic capital gain or loss.",
    "too_many_changes": "The CF modifies an overwhelming number of features at once, placing an unrealistic burden on the individual.",
    "fnlwgt_change": "The CF modifies the fnlwgt census sampling weight, which is statistical noise.",
}


def get_issue_guidance() -> str:
    """Format the taxonomy as a bullet list for injection into prompts."""
    return "\n".join(
        f"- {label}: {description}"
        for label, description in ISSUE_TAXONOMY.items()
    )
