# Issue Taxonomy for the AutoGen Courtroom Project

## Purpose

This taxonomy defines issue labels used to evaluate counterfactual explanations.

> Design goal: fewer labels, sharper definitions, minimal overlap.

---

## Taxonomy (Scored Labels)

### Quick Reference

| Label | Trigger Condition |
|---|---|
| `extreme_working_hours` | Flag if `hours_per_week` reaches an unrealistic extreme (either dangerously high, or implausibly low for a >50K income). |
| `inconsistent_work_profile` | Flag if work-related edits are internally inconsistent or temporally implausible (`workclass`, `occupation`, tenure, employment status). |
| `unactionable_capital_shift` | Flag if `capital_gain` or `capital_loss` increases to a level that is financially unrealistic given the individual's baseline profile. |
| `too_many_changes` | Flag if the counterfactual modifies an overwhelming number of features at once, placing an unrealistic burden on the individual. |

### Human-Readable Taxonomy (Grouped)

#### Work Plausibility

- **`inconsistent_work_profile`**: Work-related edits are internally inconsistent or temporally implausible. This mainly concerns changes to `workclass` and `occupation`.

#### Working Hours Realism

- **`extreme_working_hours`**: `hours_per_week` reaches an unrealistic extreme, such as dangerously high hours or implausibly low hours while still predicting an income above 50K.

#### Financial Actionability

- **`unactionable_capital_shift`**: `capital_gain` or `capital_loss` changes to a level that is financially unrealistic given the individual's baseline profile.

#### Burden and Actionability

- **`too_many_changes`**: The counterfactual modifies too many actionable features at once, placing an unrealistic burden on the individual.

---

### Mutable Feature Tiers (for `too_many_changes`)

Use the following feature tiers when counting burden:

#### Actionable Mutable Features

- `workclass`
- `occupation`
- `hours_per_week`
- `capital_gain`
- `capital_loss`

#### Frozen / Non-Actionable Features

These features are frozen during counterfactual generation and should not be counted as mutable burden targets:

- `age`
- `race`
- `sex`
- `native_country`
- `education`
- `marital_status`
- `relationship`
- `fnlwgt`
- `education_num`

If one of these frozen features changes, this should be treated as a pipeline or constraint violation, not as a scored taxonomy issue.

Notes:

- Use canonical underscored feature names in this document: `hours_per_week`, `capital_gain`, `capital_loss`, `native_country`, `education_num`, `marital_status`.
- If the raw dataset uses hyphenated names, map them before applying the taxonomy.

### Coupled-Change Heuristics (Count as One Logical Change)

To avoid over-penalizing realistic edits, some feature pairs may be treated as one logical intervention when they co-change for consistency.

Apply this heuristic during burden counting:

1. `workclass` + `occupation`: if the occupation shift is a plausible consequence of the workclass shift, count the pair as one major change.

Practical rule:

- A coupled bundle counts as one change only when the second field is explanatory or consistency-preserving rather than a separate intervention target.
- Do not collapse unrelated co-changes.

### Canonical Dictionary

```python
ISSUE_TAXONOMY = {
    "extreme_working_hours": (
        "Flag if the suggested hours_per_week reaches an unrealistic extreme, "
        "such as dangerously high hours or implausibly low hours while still "
        "predicting an income above 50K."
    ),

    "inconsistent_work_profile": (
        "Flag if work-related edits are internally inconsistent or temporally "
        "implausible, especially changes involving workclass and occupation."
    ),

    "unactionable_capital_shift": (
        "Flag if capital_gain or capital_loss changes to a level that is "
        "financially unrealistic given the individual's baseline profile."
    ),

    "too_many_changes": (
        "Flag if the counterfactual modifies too many actionable features at once, "
        "placing an unrealistic burden on the individual."
    ),
}
```

---

## Category Mapping

| Category | Labels |
|---|---|
| Work plausibility | `inconsistent_work_profile` |
| Working-hours realism | `extreme_working_hours` |
| Financial actionability | `unactionable_capital_shift` |
| Burden/actionability | `too_many_changes` |

---

## Judge Output Example

```json
{
  "overall_assessment": "problematic",
  "flagged_issues": ["extreme_working_hours", "too_many_changes"]
}
```

```json
{
  "overall_assessment": "acceptable",
  "flagged_issues": []
}
```

---

## Extension Rule

Add new labels only if they are:

1. Low-overlap with existing labels.
2. Triggered by a clear condition.
3. Measurable from available features.
