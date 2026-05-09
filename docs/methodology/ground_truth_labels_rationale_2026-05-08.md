# Ground-Truth Label Draft Rationale

> Patch date: 2026-05-08  
> Status: initial human-perspective draft for review  
> Source artifact: `annotations/ground_truth_labels.json`  
> Applied artifact: `results/cases.json`

---

## 1. Purpose

The project now needs reference labels so the three evaluation strategies can be compared:

1. metrics-only baseline
2. single-LLM evaluator
3. multi-agent debate

Because no external human-labeled dataset exists for these generated counterfactuals, this patch creates a small internal reference set for the 10 current cases.

These labels are not meant to be treated as unquestionable truth. They are a reviewable first draft: the goal is to make the rationale explicit enough that the team can agree, disagree, and revise them deliberately.

---

## 2. Method Used

For each case, I inspected:

- the original individual profile
- all generated counterfactuals
- `features_changed`
- `changes_summary`
- `cf_confidence`
- computed metrics such as sparsity and proximity
- deterministic heuristic evidence
- whether the original prediction was a false negative

I then assigned labels from the existing taxonomy only:

| Label | Human interpretation used here |
|---|---|
| `fragile_counterfactual` | At least one proposed CF barely crosses the favorable threshold, usually below `0.60` confidence. |
| `implausible_time_dependent_change` | Education or age changes violate basic time logic, especially education increasing without age increasing. |
| `too_many_changes` | The CF set contains recommendations requiring an excessive number of simultaneous interventions. |
| `unactionable_capital_shift` | The CF depends on large capital-gain / capital-loss shifts that are not realistic recommendations. |
| `extreme_working_hours` | Working-hour edits are genuinely extreme or implausible, not merely different. |
| `inconsistent_work_profile` | Workclass and occupation edits create an internally implausible work story. |

Case-level `ground_truth_issues` are the union of the per-counterfactual labels in `ground_truth_by_cf`.

---

## 3. Label Summary

| Case | Draft Ground Truth Issues |
|---:|---|
| 0 | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` |
| 1 | `fragile_counterfactual`, `implausible_time_dependent_change` |
| 2 | `fragile_counterfactual`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` |
| 3 | `unactionable_capital_shift` |
| 4 | `fragile_counterfactual`, `unactionable_capital_shift` |
| 5 | `fragile_counterfactual`, `implausible_time_dependent_change` |
| 6 | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` |
| 7 | `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes` |
| 8 | `fragile_counterfactual`, `implausible_time_dependent_change` |
| 9 | `fragile_counterfactual`, `implausible_time_dependent_change` |

---

## 4. Case-by-Case Rationale

### Case 0

Original profile: 48-year-old divorced female, `Private`, `HS-grad`, `Craft-repair`, 40 hours/week, no capital gain/loss.

Assigned labels:

- `fragile_counterfactual`
- `implausible_time_dependent_change`
- `inconsistent_work_profile`
- `too_many_changes`
- `unactionable_capital_shift`

Rationale:

- CF 0 has `cf_confidence = 0.5552`, so the favorable prediction is fragile.
- CF 1 increases education from `9` to `13` without any age increase.
- CF 3 increases education from `9` to `12` while age only increases from `48` to `50`, which is an aggressive time assumption.
- All CFs require many simultaneous interventions, often six or seven feature changes.
- Capital gain jumps from `0` to values such as `6562`, `8118`, `11808`, and `13094`.
- Several work transitions are implausible as practical advice, especially repeated shifts from `Craft-repair` to `Armed-Forces` or `Priv-house-serv` alongside workclass changes.

Why this is a strong reject case:

Even though all CFs are valid for the model, the set does not offer realistic recourse. It mixes financial jumps, occupational discontinuities, time-dependent education changes, and high burden.

---

### Case 1

Original profile: 31-year-old married male, `Private`, `Some-college`, `Adm-clerical`, 50 hours/week.

Assigned labels:

- `fragile_counterfactual`
- `implausible_time_dependent_change`

Rationale:

- CF 1 has `cf_confidence = 0.5187`.
- CF 2 has `cf_confidence = 0.5116`.
- CF 3 increases education from `10` to `13` without age increasing.
- The non-education occupation/workclass moves are not automatically implausible, so I did not label `inconsistent_work_profile`.
- The CFs are sparse enough that `too_many_changes` is not justified.

---

### Case 2

Original profile: 29-year-old female, `Local-gov`, `Bachelors`, `Prof-specialty`, 55 hours/week.

Assigned labels:

- `fragile_counterfactual`
- `inconsistent_work_profile`
- `too_many_changes`
- `unactionable_capital_shift`

Rationale:

- CF 1 has `cf_confidence = 0.5354`.
- Every CF relies on a capital-gain increase, including `3583`, `8731`, `12130`, and `12789`.
- The CFs generally require five to seven feature changes.
- CF 0 proposes `workclass = Without-pay` while still reaching a favorable income prediction through capital gain. As human-facing advice, that work profile is incoherent.
- Education changes in CF 2 and CF 3 are paired with age increases, so I did not mark `implausible_time_dependent_change`.

---

### Case 3

Original profile: 68-year-old divorced female, `State-gov`, `Masters`, `Prof-specialty`, 20 hours/week.

Assigned labels:

- `unactionable_capital_shift`

Rationale:

- All CFs depend on capital gain jumping from `0` to `10883`.
- The CFs are otherwise relatively sparse and have reasonable confidence.
- I did not label `too_many_changes` because most CFs change only two or three features.
- I did not label `fragile_counterfactual` because all confidence values are comfortably above `0.7`.

This is a good review case: the metric profile looks clean, but the recommendation still depends on a large financial event.

---

### Case 4

Original profile: 36-year-old married male, `Private`, `HS-grad`, `Craft-repair`, 40 hours/week.

Assigned labels:

- `fragile_counterfactual`
- `unactionable_capital_shift`

Rationale:

- CF 0, CF 1, and CF 3 have low confidence values: `0.5315`, `0.5838`, and `0.5124`.
- CF 1, CF 2, and CF 3 depend on capital gain increasing from `0` to `5062`.
- I did not include `extreme_working_hours` even though the heuristic flagged it for CF 3. The hour change is from `40` to `30`; from a human perspective, that is not clearly extreme enough to be a ground-truth issue by itself.
- I did not include `too_many_changes` because the CFs are relatively sparse.

This is one of the useful disagreement cases: the deterministic baseline over-flags `extreme_working_hours` relative to this draft human interpretation.

---

### Case 5

Original profile: 32-year-old married male, `Private`, `Some-college`, `Other-service`, 50 hours/week.

Assigned labels:

- `fragile_counterfactual`
- `implausible_time_dependent_change`

Rationale:

- CF 0 has `cf_confidence = 0.5673`.
- CF 1 increases education from `10` to `13` without age increasing.
- The case has only two generated CFs, and neither has enough simultaneous changes for `too_many_changes`.
- No large capital jump is present.

---

### Case 6

Original profile: 39-year-old divorced female, `Local-gov`, `Assoc-acdm`, `Other-service`, 55 hours/week.

Assigned labels:

- `implausible_time_dependent_change`
- `inconsistent_work_profile`
- `too_many_changes`
- `unactionable_capital_shift`

Rationale:

- CF 0 increases education from `12` to `16` without age increasing.
- CF 0 also proposes `workclass = Without-pay` with `occupation = Priv-house-serv`, while the favorable class is reached through large capital gain. That is not coherent recourse advice.
- All CFs require large capital-gain shifts, including `4267`, `6892`, `8619`, and `13111`.
- Most CFs require five to seven simultaneous feature changes.
- I did not include `fragile_counterfactual`; the lowest confidence is `0.6026`, which is just above the fragility threshold.

---

### Case 7

Original profile: 37-year-old female, `Private`, `Bachelors`, `Prof-specialty`, capital loss `1340`, 42 hours/week.

Assigned labels:

- `fragile_counterfactual`
- `implausible_time_dependent_change`
- `too_many_changes`

Rationale:

- CF 0 has `cf_confidence = 0.5484`.
- CF 2 has `cf_confidence = 0.5530`.
- CF 1 and CF 2 increase education without any age increase.
- All CFs require multiple coordinated interventions, including workclass, occupation or education, capital loss, and hours.
- Capital loss changes are present but not large enough to justify `unactionable_capital_shift`.

---

### Case 8

Original profile: 28-year-old married male, `Private`, `11th`, `Craft-repair`, 60 hours/week.

Important context:

- `is_false_negative = true`
- true label is `>50K`
- model prediction is `<=50K`

Assigned labels:

- `fragile_counterfactual`
- `implausible_time_dependent_change`

Rationale:

- CF 0, CF 1, and CF 2 are fragile, with confidence values `0.5321`, `0.5080`, and `0.5445`.
- CF 1 increases education from `7` to `9` without age increasing.
- The false-negative status is not a taxonomy label, but it should be mentioned by evaluators because explanations for an already misclassified person are less trustworthy.
- The CF set is sparse and close, so I did not label `too_many_changes`.

---

### Case 9

Original profile: 24-year-old married male, `Private`, `Some-college`, `Craft-repair`, 40 hours/week.

Assigned labels:

- `fragile_counterfactual`
- `implausible_time_dependent_change`

Rationale:

- All four CFs are fragile, with confidence values between `0.5014` and `0.5725`.
- CF 1 increases education from `10` to `13` without age increasing.
- Workclass/occupation/hour changes are plausible enough not to label `inconsistent_work_profile`.
- The CFs are not burdensome enough to label `too_many_changes`.

---

## 5. Expected Disagreements With Metrics-Only Baseline

After applying these labels and rerunning `run_metrics_only.py`, the metrics-only baseline scored:

| Metric | Value |
|---|---:|
| Total cases | 10 |
| Total draft ground-truth issues | 27 |
| Caught issues | 24 |
| Detection rate | 88.89% |
| Exact match rate | 60.00% |

The main disagreements are intentional and useful:

| Case | Disagreement |
|---:|---|
| 0 | Metrics-only misses `inconsistent_work_profile`. |
| 2 | Metrics-only misses `inconsistent_work_profile`. |
| 4 | Metrics-only adds `extreme_working_hours`, which I excluded as too harsh for a 40 -> 30 hour change. |
| 6 | Metrics-only misses `inconsistent_work_profile`. |

These disagreements are exactly where the LLM and multi-agent stages can show added value. A semantically aware evaluator should be able to discuss work-profile coherence and should not blindly treat every threshold-triggered hour change as a ground-truth flaw.

---

## 6. Review Questions For The Team

These are the points most worth debating:

1. Should `inconsistent_work_profile` be included for cases 0, 2, and 6?
2. Should case 4 include `extreme_working_hours`, or is 30 hours/week not extreme enough?
3. Should any large capital-gain suggestion automatically count as `unactionable_capital_shift`, or should the threshold depend on age/job/profile?
4. Should a case-level label be assigned if only one counterfactual in the set has that issue?
5. Should false-negative status become its own taxonomy label, or remain contextual evidence only?

---

## 7. Bottom Line

This label set is a pragmatic reference draft:

- It uses the existing taxonomy.
- It is grounded in actual generated counterfactuals.
- It is stricter than just copying heuristic output.
- It creates useful disagreement points for later comparison.

The next step is human review: accept, reject, or revise each label with explicit rationale.

