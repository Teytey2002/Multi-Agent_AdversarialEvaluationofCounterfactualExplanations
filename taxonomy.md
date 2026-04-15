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
| `age_reversal` | Flag if age decreases (biologically impossible). |
| `extreme_age_increase` | Flag if age increases by an extreme or unrealistic gap. |
| `race_change` | Flag if race or ethnicity changes. |
| `native_country_change` | Flag if native country changes. |
| `fnlwgt_change` | Flag if `fnlwgt` changes, since it is not a meaningful or user-actionable recommendation target. |
| `implausible_education_jump` | Flag if education changes by more than one ordinal level in one step, or to a level implausible for the rest of the profile. |
| `extreme_working_hours` | Flag if `hours_per_week` reaches an unrealistic extreme (either dangerously high, or implausibly low for a >50K income). |
| `inconsistent_work_profile` | Flag if work-related edits are internally inconsistent or temporally implausible (`workclass`, `occupation`, tenure, employment status). |
| `unactionable_capital_shift` | Flag if `capital_gain` or `capital_loss` increases to a level that is financially unrealistic given the individual's baseline profile. |
| `too_many_changes` | Flag if the counterfactual modifies an overwhelming number of features at once, placing an unrealistic burden on the individual. |
| `education_mismatch` | Optional: flag if `education` and `education_num` become inconsistent. |

### Human-Readable Taxonomy (Grouped)

#### Fairness-Sensitive

- **`sex_change`**: Sex or gender changes.
- **`race_change`**: Race or ethnicity changes.
- **`native_country_change`**: Native country changes.

#### Realism and Consistency

- **`age_reversal`**: Age decreases (biologically impossible).
- **`implausible_education_jump`**: Education changes by more than one ordinal level in one step, or to a level implausible for the rest of the profile.
- **`extreme_working_hours`**: `hours_per_week` reaches an unrealistic extreme, such as dangerously high hours (burnout risk) or implausibly low hours while predicting an income >50K.
- **`inconsistent_work_profile`**: Work-related edits are internally inconsistent or temporally implausible (`workclass`, `occupation`, tenure, employment status).
- **`education_mismatch`** (optional): `education` and `education_num` are inconsistent.

#### Burden and Actionability

- **`extreme_age_increase`**: Age increases by an extreme amount, rendering the counterfactual useless as a practical near-term plan.
- **`unactionable_capital_shift`**: `capital_gain` or `capital_loss` increases to a level that is financially unrealistic for the individual (e.g., advising a lower-income worker to simply make huge capital gains).
- **`too_many_changes`**: The counterfactual modifies an overwhelming number of features at once, placing an unrealistic burden on the individual.
- **`fnlwgt_change`**: `fnlwgt` changes, even though it is not a meaningful or user-actionable recommendation target. (Note: `fnlwgt` is a census sampling weight. If DiCE modifies this, it's a technical misconfiguration in the ML pipeline, not a semantic property of the explanation to be debated. Ideally, `fnlwgt` should be frozen or dropped before generation so the Judge doesn't waste tokens evaluating statistical noise.)

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

- Protected or fairness-sensitive: `sex`, `race`, `native_country` (Note: `age` increases are natural time progression)
- Non-actionable technical/statistical: `fnlwgt`
- Biologically impossible: Getting younger (`age` decrease)

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
    "age_reversal": "Flag if age decreases (biologically impossible).",
    "extreme_age_increase": "Flag if age increases by an extreme or unrealistic gap, making it unactionable for near-term planning.",
    "race_change": "Flag if race or ethnicity changes.",
    "native_country_change": "Flag if native country changes.",
    "fnlwgt_change": "Flag if fnlwgt changes, since it is not a meaningful or user-actionable recommendation target. Note: fnlwgt is a census sampling weight and should ideally be frozen/dropped before generation.",
    "implausible_education_jump": "Flag if education changes by more than one ordinal level in one step, or to a level implausible for the rest of the profile.",
    "extreme_working_hours": "Flag if the suggested hours_per_week reaches an unrealistic extreme (dangerously high, or implausibly low for a >50K income).",
    "inconsistent_work_profile": "Flag if work-related edits are internally inconsistent or temporally implausible (workclass, occupation, tenure, employment status).",
    "unactionable_capital_shift": "Flag if capital_gain or capital_loss increases to a level that is financially unrealistic given the individual's baseline profile.",
    "too_many_changes": "Flag if the counterfactual modifies an overwhelming number of features at once, placing an unrealistic burden on the individual.",
    "education_mismatch": "Optional: flag if education and education_num become inconsistent.",
}
```

---

## Category Mapping

| Category | Labels |
|---|---|
| Fairness-sensitive | `sex_change`, `race_change`, `native_country_change` |
| Realism/consistency | `age_reversal`, `implausible_education_jump`, `extreme_working_hours`, `inconsistent_work_profile`, `education_mismatch` (optional) |
| Burden/actionability | `extreme_age_increase`, `too_many_changes`, `fnlwgt_change`, `unactionable_capital_shift` |

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
