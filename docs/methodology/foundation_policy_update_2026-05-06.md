# Foundation Policy Update - 2026-05-06

## Why This Change Was Made

The counterfactual pipeline had several foundational assumptions that were too implicit:

- `education` and `education-num` both represented education, which duplicated signal and made the model harder to interpret.
- Keeping `education` frozen while allowing `education-num` to change created contradictory human-facing CF rows, such as `education = HS-grad` with `education-num = 13`.
- `age` and `education-num` were previously treated as frozen, even though the project discussion settled on a limited long-term recourse setting where they can increase.
- Box constraints for DiCE were fixed constants rather than values justified by the Adult dataset distribution.
- Frozen-feature violations were mixed conceptually with scored counterfactual-quality issues.
- DiCE genetic weights were manually set without a clear reason.

This update makes those decisions explicit and keeps them in one central module.

## New Feature Policy

The source of truth is now `src/policy/feature_policy.py`.

Model training excludes:

- `education`

The model keeps:

- `education-num`

DiCE may vary:

- `age`
- `education-num`
- `workclass`
- `occupation`
- `hours-per-week`
- `capital-gain`
- `capital-loss`

Frozen features:

- `fnlwgt`
- `marital-status`
- `relationship`
- `race`
- `sex`
- `native-country`

Derived display features:

- `education`

The rationale is:

- `education` is removed from training because it duplicates `education-num` and introduces categorical complexity.
- `education` is not directly mutable by DiCE and is not treated as independently frozen. It is synchronized from `education-num` after generation for display consistency.
- `age` and `education-num` are allowed only as long-term recourse features.
- Causal plausibility is checked after generation by deterministic heuristics.

## DiCE Generation Changes

`src/pipeline/generate_cf.py` now reads the policy from `policy.feature_policy`.

The DiCE genetic configuration uses the library's reference/default values:

| Parameter | Value |
|---|---:|
| `proximity_weight` | `0.2` |
| `sparsity_weight` | `0.2` |
| `diversity_weight` | `5.0` |
| `categorical_penalty` | `0.1` |
| `stopping_threshold` | `0.5` |
| `posthoc_sparsity_param` | `0.1` |
| `posthoc_sparsity_algorithm` | `binary` |

The generated policy metadata is saved to:

- `results/generation_policy.json`

That file records the policy, DiCE parameters, features allowed to vary, frozen features, derived display features, causal checks, and per-instance permitted ranges.

## Education Display Synchronization

The Adult dataset has both:

- `education`: categorical label, such as `HS-grad` or `Bachelors`
- `education-num`: ordinal education level, such as `9` or `13`

The model now uses only `education-num`, but result artifacts still keep `education` because it is easier for humans to read.

To avoid inconsistent CF explanations, generated rows now synchronize `education` from `education-num` using the Adult mapping:

| `education-num` | `education` |
|---:|---|
| 1 | `Preschool` |
| 2 | `1st-4th` |
| 3 | `5th-6th` |
| 4 | `7th-8th` |
| 5 | `9th` |
| 6 | `10th` |
| 7 | `11th` |
| 8 | `12th` |
| 9 | `HS-grad` |
| 10 | `Some-college` |
| 11 | `Assoc-voc` |
| 12 | `Assoc-acdm` |
| 13 | `Bachelors` |
| 14 | `Masters` |
| 15 | `Prof-school` |
| 16 | `Doctorate` |

Example:

```text
education-num: 9 -> 13
education:     HS-grad -> Bachelors
```

This is counted as one education intervention through `education-num`; the synchronized display-label change is not counted as a separate feature change.

## Box Constraints

`src/pipeline/explore_data.py` now adds numerical percentiles to `results/feature_catalog.json`.

`feature_policy.build_permitted_range()` now derives per-instance ranges from:

- the individual's current value
- empirical dataset ranges and percentiles
- configured maximum deltas for long-term changes

This replaces the previous fixed constants such as `hours-per-week: [20, 50]` and `capital-gain: [0, 5000]`.

## Heuristic And Taxonomy Changes

`src/policy/heuristics.py` now uses the centralized feature policy.

New scored issue label:

- `implausible_time_dependent_change`

This is flagged when:

- `age` decreases
- `education_num` decreases
- `education_num` increases without age increasing
- `education_num` increases more than the age increase can plausibly support
- `age` or `education_num` becomes non-integer

Frozen-feature changes remain separate:

- They are recorded as `constraint_violations`.
- They are not scored issue labels.
- The Judge may mention them in reasoning but should not include them in `flagged_issues`.

Unsynchronized `education` changes are also recorded as constraint violations because `education` may only change to match `education-num`.

## Rebuilt Artifacts

After the code changes, the pipeline was rerun through:

1. `python -m pipeline.explore_data`
2. `python -m pipeline.train`
3. `python -m pipeline.predict`
4. `python -m pipeline.generate_cf`
5. `python -m pipeline.cf_metrics`
6. `python -m pipeline.case_builder --pretty`

Updated artifacts include:

- `models/logistic_regression.joblib`
- `results/logistic_regression_metrics.json`
- `results/unfavorable_samples.csv`
- `results/counterfactuals.csv`
- `results/generation_policy.json`
- `results/cf_metrics_per_instance.csv`
- `results/cf_metrics_global.csv`
- `results/cases.json`
- `results/feature_catalog.json`

The rebuilt model uses 13 input features after removing raw `education`.

Latest model metrics:

| Metric | Value |
|---|---:|
| Accuracy | `0.8523` |
| Precision | `0.7379` |
| Recall | `0.5937` |
| F1 | `0.6580` |

Latest global counterfactual metrics:

| Metric | Value |
|---|---:|
| Validity mean | `1.0` |
| Continuous proximity mean | `-0.6424` |
| Categorical proximity mean | `0.7552` |
| Sparsity mean | `0.7179` |
| Continuous diversity mean | `0.5138` |
| Categorical diversity mean | `0.2479` |
| Count diversity mean | `0.2833` |

## Validation

The new focused test suite is:

- `tests/test_foundation_policy.py`

It checks:

- `education` is excluded from model inputs.
- `education-num` remains in the model.
- DiCE genetic defaults are used.
- empirical permitted ranges are non-decreasing for `age` and `education-num`.
- `education` labels are synchronized from `education-num`.
- synchronized `education` label changes are not counted as independent CF changes.
- unsynchronized `education` label changes are constraint violations.
- time-dependent education changes are scored issues, not frozen-feature violations.
- frozen protected changes remain constraint violations.
- the new taxonomy label is accepted by verdict parsing.

Validation commands run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m unittest discover -s tests
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall src tests
```

Both passed.

Additional artifact sanity checks:

- `results/counterfactuals.csv`: 47 rows checked, 0 `education` / `education-num` mismatches.
- `results/cases.json`: 10 cases checked, 0 independent `education` feature changes after synchronization.

## Remaining Decision

The pipeline currently keeps all DiCE counterfactuals and flags implausible ones for the agents.

The next design decision is whether to:

1. keep this flag-only behavior, so the evaluator agents judge the generated set, or
2. filter/reject implausible counterfactuals before they reach the LLM stage.

For a course project, the current flag-only behavior is defensible because it preserves evidence of generator failure and lets the adversarial evaluation layer do its job.
