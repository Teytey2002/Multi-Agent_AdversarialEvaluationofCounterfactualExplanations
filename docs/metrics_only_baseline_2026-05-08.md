# Metrics-Only Baseline

> Patch date: 2026-05-08  
> Status: implemented, tested, and runnable  
> Scope: deterministic non-LLM evaluation baseline for generated counterfactual cases

---

## 1. Why This Patch Exists

The project compares three evaluation strategies for counterfactual explanations:

| Strategy | Question Answered | Uses LLM? |
|---|---|---:|
| Metrics-only baseline | What would a deterministic rule system conclude from the computed evidence? | No |
| Single-LLM evaluator | What would one LLM conclude from the same evidence? | Yes |
| Multi-agent debate | Does adversarial discussion improve the final judgment? | Yes |

Before this patch, the project already computed counterfactual metrics and heuristic issue flags, but those outputs were still evidence, not an evaluation method.

The missing step was a deterministic evaluator that turns the evidence into the same type of verdict expected from the LLM Judge:

```json
{
  "overall_assessment": "fair | ambiguous | unfair",
  "flagged_issues": ["issue_label"],
  "severity": "low | medium | high",
  "recommended_action": "accept | review | reject"
}
```

This patch adds that missing non-LLM competitor.

Important distinction:

- The metrics-only baseline is not ground truth.
- It is one evaluated system.
- Human/team annotations in `ground_truth_issues` remain the future reference labels.

---

## 2. What Was Added

### New evaluator package

`src/evaluators/`

| File | Purpose |
|---|---|
| `src/evaluators/__init__.py` | Exposes deterministic evaluator functions. |
| `src/evaluators/metrics_only.py` | Converts computed metrics and heuristic evidence into Judge-compatible verdicts. |

### New CLI runner

`src/run_metrics_only.py`

This script loads `results/cases.json`, applies the deterministic evaluator, and writes structured results under:

```text
results/metrics_only_outputs/
```

That output directory is ignored by git because it is regenerated, just like debate outputs.

### New tests

`tests/test_metrics_only.py`

The tests cover:

- clean cases becoming `fair / accept`
- fragile-only cases becoming `ambiguous / review`
- multiple critical issues becoming `unfair / reject`
- constraint violations staying separate from scored issue labels
- fallback to per-counterfactual heuristic evidence

Validation run:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m unittest discover -s tests
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m compileall src tests
```

Result:

```text
14 tests passed
compileall passed
```

---

## 3. Where It Fits In The Pipeline

The current project flow is now:

```text
OpenML Adult dataset
        |
        v
train.py
        |
        v
predict.py
        |
        v
generate_cf.py
        |
        v
cf_metrics.py
        |
        v
case_builder.py
        |
        +--> run_metrics_only.py   deterministic baseline
        |
        +--> run_debate.py         single-LLM or multi-agent evaluation
```

Both branches consume the same `results/cases.json`.

That design matters because the comparison is controlled: the metrics-only baseline, the single LLM, and the multi-agent debate all start from the same computed evidence.

---

## 4. Input Evidence Used By The Baseline

The baseline consumes case objects from `results/cases.json`.

For each case, it reads:

| Evidence | Source Field | Meaning |
|---|---|---|
| Heuristic issue labels | `heuristic_summary.flagged_issues_union` | Deterministic taxonomy-aligned issue labels. |
| Constraint violations | `heuristic_summary.constraint_violations_union` | Pipeline/policy problems, not scored issue labels. |
| Counterfactual metrics | `metrics` | Validity, proximity, sparsity, diversity. |
| False-negative marker | `is_false_negative` | Whether the model predicted unfavorable although the true label is favorable. |
| Per-CF fallback evidence | `counterfactuals[*].heuristic_metrics` | Used if case-level summary is missing. |

The baseline does not inspect the raw model directly and does not call any LLM API.

---

## 5. Deterministic Decision Logic

The implementation lives in:

```text
src/evaluators/metrics_only.py
```

### 5.1 Scored Issues

The evaluator accepts only taxonomy labels from `agents.prompts.get_valid_issue_labels()`.

Current scored labels include:

```text
fragile_counterfactual
implausible_time_dependent_change
too_many_changes
unactionable_capital_shift
extreme_working_hours
inconsistent_work_profile
```

### 5.2 Critical Issues

The baseline treats these as critical:

```python
CRITICAL_ISSUES = {
    "too_many_changes",
    "unactionable_capital_shift",
    "implausible_time_dependent_change",
    "extreme_working_hours",
    "inconsistent_work_profile",
}
```

Interpretation:

- one critical issue -> usually `medium` severity
- two or more critical issues -> `high` severity
- any constraint violation -> `high` severity
- fragile-only findings -> usually `low` severity

### 5.3 Metric Warning Thresholds

The evaluator also records non-taxonomy metric warnings:

```python
METRIC_WARNING_THRESHOLDS = {
    "validity_min": 1.0,
    "sparsity_low": 0.65,
    "continuous_proximity_low": -1.0,
    "categorical_proximity_low": 0.70,
}
```

These warnings support the rationale but are not added to `flagged_issues`.

Example:

```json
"metric_warnings": [
  "low_sparsity",
  "low_continuous_proximity",
  "low_categorical_proximity"
]
```

This separation is intentional:

- `flagged_issues` stays aligned with the taxonomy.
- `metric_warnings` explain why the deterministic baseline was cautious.

---

## 6. Output Schema

The metrics-only output mirrors the LLM Judge schema.

Example:

```json
{
  "case_id": 3,
  "overall_assessment": "ambiguous",
  "flagged_issues": ["unactionable_capital_shift"],
  "severity": "medium",
  "confidence": 0.85,
  "reasoning_summary": "deterministic heuristics flagged unactionable_capital_shift.",
  "recommended_action": "review"
}
```

This makes later comparison direct:

| Field | Metrics-Only | Single LLM | Multi-Agent |
|---|---:|---:|---:|
| `overall_assessment` | Yes | Yes | Yes |
| `flagged_issues` | Yes | Yes | Yes |
| `severity` | Yes | Yes | Yes |
| `recommended_action` | Yes | Yes | Yes |
| `reasoning_summary` | Short deterministic summary | LLM explanation | Judge synthesis |

---

## 7. How To Run It

From the repository root:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe src\run_metrics_only.py
```

Run selected cases:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe src\run_metrics_only.py --case-ids 0 3 8
```

Outputs:

```text
results/metrics_only_outputs/metrics_only_<timestamp>/metrics_only_results.json
results/metrics_only_outputs/metrics_only_latest.json
```

---

## 8. Current Run Summary

The baseline was run on the current `results/cases.json` at:

```text
2026-05-08T11:36:52
```

Current artifact:

```text
results/metrics_only_outputs/metrics_only_latest.json
```

### 8.1 Verdict Distribution

| Case | Assessment | Severity | Action | Main Issues |
|---:|---|---|---|---|
| 0 | unfair | high | reject | fragile, time-dependent, too many changes, capital shift |
| 1 | ambiguous | medium | review | fragile, time-dependent |
| 2 | unfair | high | reject | fragile, too many changes, capital shift |
| 3 | ambiguous | medium | review | capital shift |
| 4 | unfair | high | reject | extreme hours, fragile, capital shift |
| 5 | ambiguous | medium | review | fragile, time-dependent |
| 6 | unfair | high | reject | time-dependent, too many changes, capital shift |
| 7 | unfair | high | reject | fragile, time-dependent, too many changes |
| 8 | ambiguous | medium | review | fragile, time-dependent |
| 9 | ambiguous | medium | review | fragile, time-dependent |

Summary:

| Category | Count |
|---|---:|
| `unfair / reject` | 5 |
| `ambiguous / review` | 5 |
| `fair / accept` | 0 |

Interpretation:

The deterministic baseline is conservative on the current generated counterfactuals. It finds every case at least review-worthy because the generated set commonly contains fragile flips, time-dependent education/age concerns, large capital-gain recommendations, or high feature-change burden.

---

## 9. Concrete Case Interpretations

### Case 0: High-severity reject

Original profile excerpt:

| Field | Value |
|---|---|
| Age | 48 |
| Education | HS-grad |
| Occupation | Craft-repair |
| Hours per week | 40 |
| Capital gain | 0 |
| Capital loss | 0 |
| Prediction confidence | 0.9735 |
| False negative | False |

Counterfactual quality metrics:

| Metric | Value |
|---|---:|
| Validity | 1.0 |
| Continuous proximity | -1.6527 |
| Categorical proximity | 0.6563 |
| Sparsity | 0.4821 |
| Count diversity | 0.4881 |

Metrics-only verdict:

```json
{
  "overall_assessment": "unfair",
  "flagged_issues": [
    "fragile_counterfactual",
    "implausible_time_dependent_change",
    "too_many_changes",
    "unactionable_capital_shift"
  ],
  "severity": "high",
  "recommended_action": "reject"
}
```

Why the baseline rejects it:

- CF 0 changes six features: age, workclass, occupation, capital gain, capital loss, and hours.
- CF 2 and CF 3 change seven features.
- Capital gain jumps from `0` to values such as `11808` and `13094`.
- One CF only reaches `cf_confidence = 0.5552`, which is fragile.
- Sparsity is low at `0.4821`, meaning the recommendations require many simultaneous edits.

Example CF changes:

```json
{
  "cf_rank": 2,
  "cf_confidence": 0.8805,
  "changes_summary": {
    "age": {"from": 48, "to": 53},
    "workclass": {"from": "Private", "to": "Local-gov"},
    "education_num": {"from": 9, "to": 12},
    "occupation": {"from": "Craft-repair", "to": "Armed-Forces"},
    "capital_gain": {"from": 0, "to": 11808},
    "capital_loss": {"from": 0, "to": 1040},
    "hours_per_week": {"from": 40, "to": 47}
  }
}
```

How this will be exploited later:

The LLM or multi-agent system should be able to discuss why the same evidence is problematic, rather than merely repeat the labels. A strong multi-agent answer should explain the burden, realism, and financial actionability issues in human terms.

---

### Case 3: Medium-severity review

Original profile excerpt:

| Field | Value |
|---|---|
| Age | 68 |
| Education | Masters |
| Occupation | Prof-specialty |
| Hours per week | 20 |
| Capital gain | 0 |
| Prediction confidence | 0.8989 |
| False negative | False |

Counterfactual quality metrics:

| Metric | Value |
|---|---:|
| Validity | 1.0 |
| Continuous proximity | -0.2434 |
| Categorical proximity | 0.8438 |
| Sparsity | 0.8393 |
| Count diversity | 0.1071 |

Metrics-only verdict:

```json
{
  "overall_assessment": "ambiguous",
  "flagged_issues": ["unactionable_capital_shift"],
  "severity": "medium",
  "recommended_action": "review"
}
```

Why the baseline reviews it rather than rejects it:

- The CFs are relatively sparse.
- Proximity is reasonable compared with case 0.
- The main recurring issue is a large capital gain jump.

Example CF:

```json
{
  "cf_rank": 0,
  "cf_confidence": 0.8885,
  "changes_summary": {
    "workclass": {"from": "State-gov", "to": "Federal-gov"},
    "capital_gain": {"from": 0, "to": 10883}
  }
}
```

Interpretation:

The deterministic baseline cannot decide whether this is semantically acceptable. It sees a sparse, valid CF set, but the recommendation repeatedly relies on a large financial shift. That is exactly the type of case where the LLM or debate system may add value by explaining whether the issue should dominate the final judgment.

---

### Case 8: False-negative caution

Original profile excerpt:

| Field | Value |
|---|---|
| Age | 28 |
| Education | 11th |
| Occupation | Craft-repair |
| Hours per week | 60 |
| Prediction confidence | 0.6376 |
| False negative | True |

Counterfactual quality metrics:

| Metric | Value |
|---|---:|
| Validity | 1.0 |
| Continuous proximity | -0.0833 |
| Categorical proximity | 0.8125 |
| Sparsity | 0.8750 |
| Count diversity | 0.1786 |

Metrics-only verdict:

```json
{
  "overall_assessment": "ambiguous",
  "flagged_issues": [
    "fragile_counterfactual",
    "implausible_time_dependent_change"
  ],
  "severity": "medium",
  "recommended_action": "review"
}
```

Example CFs:

```json
[
  {
    "cf_rank": 0,
    "cf_confidence": 0.5321,
    "changes_summary": {
      "occupation": {"from": "Craft-repair", "to": "Exec-managerial"}
    }
  },
  {
    "cf_rank": 1,
    "cf_confidence": 0.5080,
    "changes_summary": {
      "education_num": {"from": 7, "to": 9}
    }
  }
]
```

Interpretation:

This case is useful for the LLM stage because the metrics look superficially good: high sparsity and close proximity. But the original prediction is a false negative, and the CF confidences are barely above the threshold. A strong evaluator should not just say "sparse therefore good"; it should notice that the model may already be wrong for this person.

---

## 10. Current Global Metric Context

From `results/cf_metrics_global.csv`:

| Metric | Mean |
|---|---:|
| Validity | 1.0000 |
| Continuous proximity | -0.6424 |
| Categorical proximity | 0.7552 |
| Sparsity | 0.7179 |
| Continuous diversity | 0.5138 |
| Categorical diversity | 0.2479 |
| Count diversity | 0.2833 |

Interpretation:

- Validity is perfect: all generated CFs achieve the desired class.
- That does not mean they are good recommendations.
- The baseline exposes this distinction by flagging cases where valid CFs still depend on fragile confidence, large financial jumps, implausible time logic, or excessive burden.

This is central to the project: validity alone is not enough.

---

## 11. Why This Helps Prove Added Value Later

The metrics-only baseline establishes a clear lower-complexity comparator.

Later, the project can ask:

| Comparison | Scientific Question |
|---|---|
| Metrics-only vs human labels | How far can deterministic rules go? |
| Single LLM vs metrics-only | Does semantic reasoning improve over fixed thresholds? |
| Multi-agent vs single LLM | Does adversarial debate improve issue detection, severity calibration, or reasoning quality? |
| Multi-agent vs metrics-only | Does the added complexity produce better final judgments? |

The baseline gives the project an anchor:

- If the LLM systems merely reproduce the metrics-only verdicts, their added value is weak.
- If they correct over-conservative rule decisions, the LLM adds contextual judgment.
- If the debate catches issues missed by the single LLM, the multi-agent design adds value.
- If the debate produces clearer reasoning while preserving accuracy, that is also evidence of value.

---

## 12. How To Compare Later

Once `ground_truth_issues` is filled manually, all systems can be scored with the same machinery:

```text
ground_truth_issues
        |
        +--> compare with metrics-only flagged_issues
        +--> compare with single-LLM flagged_issues
        +--> compare with multi-agent flagged_issues
```

Quantitative metrics:

| Metric | Meaning |
|---|---|
| Issue precision | Of the issues flagged by a system, how many were in the reference labels? |
| Issue recall | Of the reference issues, how many did the system catch? |
| Issue F1 | Balanced precision/recall score. |
| Exact match | Did the system predict exactly the same issue set? |
| False positives | Did the system invent issues not present in the reference labels? |

Qualitative comparison:

| Dimension | What To Look For |
|---|---|
| Faithfulness | Does the reasoning cite actual case values? |
| Severity calibration | Does the system overstate or understate issues? |
| Actionability analysis | Does it explain whether the person could realistically act on the CF? |
| Robustness | Does it notice fragile confidence near 0.5? |
| Debate value | Does the Judge synthesize opposing arguments better than a single evaluator? |

---

## 13. Current Limitation Of The Baseline Output

The current `metrics_only_latest.json` includes agreement metrics such as:

```json
{
  "false_positive_rate": 100.0,
  "exact_match_rate": 0.0
}
```

These numbers should not be interpreted yet.

Reason:

```json
"ground_truth_issues": []
```

is still empty for every case in `results/cases.json`.

So every flagged issue currently appears as an "extra" issue. This is expected. It does not mean the baseline is wrong; it means the reference labels have not been created.

Correct interpretation today:

- Use the verdicts as deterministic baseline outputs.
- Do not use the agreement scores until manual/team labels exist.

---

## 14. Strengths And Weaknesses

### Strengths

| Strength | Why It Matters |
|---|---|
| Reproducible | Same input cases always produce the same verdicts. |
| Cheap | No API calls or model latency. |
| Transparent | Rules and thresholds are visible in `metrics_only.py`. |
| Comparable | Output schema matches the LLM Judge schema. |
| Grounded | Uses computed pipeline evidence, not generated prose. |

### Weaknesses

| Weakness | Consequence |
|---|---|
| Rule-bound | Cannot reason beyond encoded thresholds. |
| Conservative | May over-flag borderline cases. |
| No semantic nuance | Cannot judge whether a work transition is contextually plausible unless already encoded. |
| No human reference | Cannot prove correctness until `ground_truth_issues` is filled. |
| Limited explanation quality | Summaries are short and mechanical by design. |

---

## 15. Recommended Next Step

The most solid next step is not to tune the baseline blindly.

Instead:

1. Keep the baseline deterministic and transparent.
2. Manually annotate `ground_truth_issues` for the 10 current cases.
3. Run:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe src\run_metrics_only.py
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe src\run_debate.py --single-llm
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe src\run_debate.py
```

4. Compare all three systems against the same reference labels.
5. Use qualitative analysis to explain where the LLM and multi-agent systems go beyond the deterministic baseline.

This gives the final project a defensible experimental structure:

```text
Human/team reference labels
        |
        +--> metrics-only baseline
        +--> single-LLM baseline
        +--> multi-agent proposed method
        |
        v
quantitative comparison + qualitative reasoning analysis
```

---

## 16. Bottom Line

The metrics-only baseline is now a complete first comparator.

It proves what can already be detected from fixed metrics and heuristics. The future LLM and multi-agent stages must justify themselves by doing more than reproducing those deterministic labels: they should improve severity calibration, explain trade-offs, identify contextual plausibility, and produce better human-facing judgments.

