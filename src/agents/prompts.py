"""
Placeholder issue taxonomy for counterfactual evaluation.

>>> IVAN: Replace this file with your finalised taxonomy. <<<

Each entry maps an issue label (used by agents in verdicts) to a short
human-readable description.  Agents receive these labels in their system
prompts so they share a common vocabulary.

Guidelines for the taxonomy:
- Labels should be snake_case identifiers.
- Descriptions should be one sentence, ≤ 20 words.
- Keep the dict ordered from most to least severe.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Default taxonomy — covers common CF quality issues.
# Replace with Ivan's research-grounded taxonomy when available.
# ---------------------------------------------------------------------------

ISSUE_TAXONOMY: dict[str, str] = {
    "immutable_feature_change": (
        "The CF changes a feature the individual cannot realistically alter "
        "(e.g. age, race, sex, native-country)."
    ),
    "proxy_feature_change": (
        "The CF changes a feature that may act as a proxy for a protected "
        "attribute (e.g. marital-status as proxy for age/gender)."
    ),
    "unrealistic_change": (
        "The CF requires an implausibly large jump in a feature value "
        "(e.g. capital-gain from 0 to the maximum)."
    ),
    "non_actionable_change": (
        "The CF changes a feature that is outside the individual's control "
        "or extremely difficult to change."
    ),
    "too_many_changes": (
        "The CF modifies too many features at once, making the suggestion "
        "impractical or hard to interpret."
    ),
    "low_confidence_cf": (
        "The CF barely crosses the decision boundary — the model's "
        "confidence in the flipped class is very low."
    ),
    "inconsistent_changes": (
        "The CF contains internally contradictory feature changes "
        "(e.g. switching to a high-skill job while lowering hours)."
    ),
}


def get_issue_guidance() -> str:
    """Format the taxonomy as a bullet list for injection into prompts."""
    return "\n".join(
        f"- {label}: {description}"
        for label, description in ISSUE_TAXONOMY.items()
    )
