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
| `implausible_time_dependent_change` | Flag if `age` or `education_num` changes violate time logic, such as age decreasing or education increasing without enough age increase. |
| `unactionable_capital_shift` | Flag if `capital_gain` or `capital_loss` increases to a level that is financially unrealistic given the individual's baseline profile. |
| `too_many_changes` | Flag if the counterfactual modifies an overwhelming number of features at once, placing an unrealistic burden on the individual. |
| `fragile_counterfactual` | Flag if the counterfactual barely reaches the favorable class, with `cf_confidence` close to the decision threshold. |

### Human-Readable Taxonomy (Grouped)

#### Work Plausibility

- **`inconsistent_work_profile`**: Work-related edits are internally inconsistent or temporally implausible. This mainly concerns changes to `workclass` and `occupation`.

#### Time-Dependent Plausibility

- **`implausible_time_dependent_change`**: `age` or `education_num` changes violate basic time logic. Age must not decrease; education level must not decrease; education level must not increase unless age also increases by a reasonable amount. Non-integer `age` or `education_num` values are also implausible because these fields represent discrete quantities in the Adult dataset.

#### Working Hours Realism

- **`extreme_working_hours`**: `hours_per_week` reaches an unrealistic extreme, such as dangerously high hours or implausibly low hours while still predicting an income above 50K.

#### Financial Actionability

- **`unactionable_capital_shift`**: `capital_gain` or `capital_loss` changes to a level that is financially unrealistic given the individual's baseline profile.

#### Burden and Actionability

- **`too_many_changes`**: The counterfactual modifies too many actionable features at once, placing an unrealistic burden on the individual.

#### Prediction Robustness

- **`fragile_counterfactual`**: The counterfactual technically reaches the favorable class, but only barely. This means the recommendation is fragile because a small model perturbation could make it unfavorable again.

---

### Mutable Feature Tiers (for `too_many_changes`)

Use the following feature tiers when counting burden:

#### Actionable Mutable Features

- `age` (increase only, bounded horizon)
- `education_num` (increase only, must be coupled with sufficient age increase)
- `workclass`
- `occupation`
- `hours_per_week`
- `capital_gain`
- `capital_loss`

#### Frozen / Non-Actionable Features

These features are frozen during counterfactual generation and should not be counted as mutable burden targets:

- `race`
- `sex`
- `native_country`
- `marital_status`
- `relationship`
- `fnlwgt`

If one of these frozen features changes, this should be treated as a pipeline or constraint violation, not as a scored taxonomy issue.

#### Derived Display Features

- `education`

`education` is excluded from model training and is not directly mutable by DiCE. It is synchronized from `education_num` after generation. If `education_num` changes from `9` to `13`, the displayed `education` label should change from `HS-grad` to `Bachelors`, and this should count as one education intervention rather than two independent feature changes.

Notes:

- Use canonical underscored feature names in this document: `hours_per_week`, `capital_gain`, `capital_loss`, `native_country`, `education_num`, `marital_status`.
- If the raw dataset uses hyphenated names, map them before applying the taxonomy.

### Coupled-Change Heuristics (Count as One Logical Change)

To avoid over-penalizing realistic edits, some feature pairs may be treated as one logical intervention when they co-change for consistency.

Apply this heuristic during burden counting:

1. `workclass` + `occupation`: if the occupation shift is a plausible consequence of the workclass shift, count the pair as one major change.
2. `age` + `education_num`: if education increases and age increases enough to make the change temporally plausible, count the pair as one long-term intervention.

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

    "implausible_time_dependent_change": (
        "Flag if age or education_num changes violate basic time logic, such as "
        "age decreasing, education_num decreasing, or education_num increasing "
        "without enough age increase."
    ),

    "unactionable_capital_shift": (
        "Flag if capital_gain or capital_loss changes to a level that is "
        "financially unrealistic given the individual's baseline profile."
    ),

    "too_many_changes": (
        "Flag if the counterfactual modifies too many actionable features at once, "
        "placing an unrealistic burden on the individual."
    ),

    "fragile_counterfactual": (
        "Flag if the counterfactual barely reaches the favorable class, "
        "with cf_confidence close to the decision threshold."
    ),
}
```

---

## Category Mapping

| Category | Labels |
|---|---|
| Work plausibility | `inconsistent_work_profile` |
| Time-dependent plausibility | `implausible_time_dependent_change` |
| Working-hours realism | `extreme_working_hours` |
| Financial actionability | `unactionable_capital_shift` |
| Burden/actionability | `too_many_changes` |
| Prediction robustness | `fragile_counterfactual` |
---

## Judge Output Example

```json
{
  "overall_assessment": "unfair",
  "flagged_issues": ["extreme_working_hours", "too_many_changes"]
}
```

```json
{
  "overall_assessment": "fair",
  "flagged_issues": []
}
```

---

## Extension Rule

Add new labels only if they are:

1. Low-overlap with existing labels.
2. Triggered by a clear condition.
3. Measurable from available features.
