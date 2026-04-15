# Issue Taxonomy for the AutoGen Courtroom Project

## Purpose

This taxonomy defines issue labels used to evaluate counterfactual explanations.

> Design goal: fewer labels, sharper definitions, minimal overlap.

---

## Taxonomy (Scored Labels)

### Quick Reference

| Label | Trigger Condition |
|---|---|
| `sex_change` | Flag if sex or gender changes. |
| `age_change` | Flag if age changes. |
| `race_change` | Flag if race or ethnicity changes. |
| `native_country_change` | Flag if native country changes. |
| `fnlwgt_change` | Flag if `fnlwgt` changes, since it is not a meaningful or user-actionable recommendation target. |
| `implausible_education_jump` | Flag if education changes by more than one ordinal level in one step, or to a level implausible for the rest of the profile. |
| `implausible_hours_increase` | Flag if `hours_per_week` increases by >20 absolute hours OR >50% relative. |
| `inconsistent_work_profile` | Flag if work-related edits are internally inconsistent or temporally implausible (`workclass`, `occupation`, tenure, employment status). |
| `too_many_changes` | Flag if >=3 major mutable features change OR 2 major + >=2 minor mutable features change; numeric features count as changed at >=10% relative change or >=0.5 z-score shift. |
| `education_mismatch` | Optional: flag if `education` and `education_num` become inconsistent. |

### Human-Readable Taxonomy (Grouped)

#### Fairness-Sensitive

- **`sex_change`**: Sex or gender changes.
- **`age_change`**: Age changes.
- **`race_change`**: Race or ethnicity changes.
- **`native_country_change`**: Native country changes.

#### Realism and Consistency

- **`implausible_education_jump`**: Education changes by more than one ordinal level in one step, or to a level implausible for the rest of the profile.
- **`implausible_hours_increase`**: `hours_per_week` increases by more than 20 absolute hours or more than 50% relative.
- **`inconsistent_work_profile`**: Work-related edits are internally inconsistent or temporally implausible (`workclass`, `occupation`, tenure, employment status).
- **`education_mismatch`** (optional): `education` and `education_num` are inconsistent.

#### Burden and Actionability

- **`too_many_changes`**: At least 3 major mutable features change, or 2 major plus at least 2 minor mutable features change; numeric features count as changed at >=10% relative change or >=0.5 z-score shift.
- **`fnlwgt_change`**: `fnlwgt` changes, even though it is not a meaningful or user-actionable recommendation target.

### Mutable Feature Tiers (for `too_many_changes`)

Use the following feature tiers when counting burden:

#### Major Mutable Features

- `education`
- `hours_per_week`
- `workclass`
- `occupation`
- `marital_status`
- `capital_gain`
- `capital_loss`

#### Minor Mutable Features

- `relationship`
- `education_num` (derived/paired with `education`)
- `workclass`-adjacent status fields (if represented separately, e.g., tenure/employment flags)

#### Not Actionable or Protected (Do Not Count as Mutable Burden Targets)

- Protected or fairness-sensitive: `sex`, `age`, `race`, `native_country`
- Non-actionable technical/statistical: `fnlwgt`

Notes:

- If your exact schema differs, keep the same tiering principle and map local feature names to these groups.
- Numeric features are counted as changed when the change is at least 10% relative or at least 0.5 standard deviations from baseline.

### Coupled-Change Heuristics (Count as One Logical Change)

To avoid over-penalizing realistic edits, some feature pairs should be treated as one logical intervention when they co-change for consistency.

Apply these heuristics during burden counting:

1. `education` + `education_num`: if both change in the same direction and remain semantically aligned, count as one major change.
2. `workclass` + `occupation`: if the occupation shift is a plausible consequence of the workclass shift, count as one major change.
3. Work-status pairings (for schemas with tenure/employment flags): if status fields co-update only to keep internal consistency, count the bundle as one change.
4. Do not collapse unrelated co-changes: if two fields change without a clear dependency, count them separately.

Practical rule:

- A coupled bundle counts as one change only when the second field is explanatory/consistency-preserving rather than a separate intervention target.

### Canonical Dictionary

```python
ISSUE_TAXONOMY = {
    "sex_change": "Flag if sex or gender changes.",
    "age_change": "Flag if age changes.",
    "race_change": "Flag if race or ethnicity changes.",
    "native_country_change": "Flag if native country changes.",
    "fnlwgt_change": "Flag if fnlwgt changes, since it is not a meaningful or user-actionable recommendation target.",
    "implausible_education_jump": "Flag if education changes by more than one ordinal level in one step, or to a level implausible for the rest of the profile.",
    "implausible_hours_increase": "Flag if hours_per_week increases by >20 absolute hours OR >50% relative.",
    "inconsistent_work_profile": "Flag if work-related edits are internally inconsistent or temporally implausible (workclass, occupation, tenure, employment status).",
    "too_many_changes": "Flag if >=3 major mutable features change OR 2 major + >=2 minor mutable features change; numeric features count as changed at >=10% relative change or >=0.5 z-score shift.",
    "education_mismatch": "Optional: flag if education and education_num become inconsistent.",
}
```

---

## Category Mapping

| Category | Labels |
|---|---|
| Fairness-sensitive | `sex_change`, `age_change`, `race_change`, `native_country_change` |
| Realism/consistency | `implausible_education_jump`, `implausible_hours_increase`, `inconsistent_work_profile`, `education_mismatch` (optional) |
| Burden/actionability | `too_many_changes`, `fnlwgt_change` |

---

## Judge Output Example

```json
{
  "overall_assessment": "unfair",
  "flagged_issues": ["sex_change", "too_many_changes"]
}
```

---

## Extension Rule

Add new labels only if they are:

1. Low-overlap with existing labels.
2. Triggered by a clear condition.
3. Measurable from available features.
