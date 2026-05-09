# Single LLM vs Metrics-Only Baseline

Date: 2026-05-08  
Dataset artifact: `results/cases.json`  
Reference labels: `annotations/ground_truth_labels.json`

This document records the first complete single-LLM evaluation run after the
Groq-only cleanup and compares it against the deterministic metrics-only
baseline.

---

## 1. What Changed Before Running

The single-LLM system was corrected so the model no longer receives the draft
reference labels in its prompt.

Relevant code:

| File | Role |
|---|---|
| `src/agents/debate.py` | Builds the prompt-safe case payload for both single-LLM and multi-agent runs. |
| `src/agents/config.py` | Resolves Groq-only LLM configuration. Non-Groq providers now fail explicitly. |
| `scripts/run_debate.py` | Runs either `--single-llm` or multi-agent debate and scores the verdict after generation. |
| `tests/test_single_llm_prompt_config.py` | Regression test proving ground-truth fields are excluded from the LLM prompt. |

The fields `ground_truth_issues`, `ground_truth_by_cf`, and
`ground_truth_source` remain in `results/cases.json`, but they are used only
after the LLM has produced a verdict.

---

## 2. Groq Configuration

The project now uses Groq only.

```powershell
$env:PYTHONPATH="src"; python scripts/run_debate.py --single-llm --max-tokens 400
```

Run configuration:

| Field | Value |
|---|---|
| Provider | `groq` |
| Model | `llama-3.1-8b-instant` |
| Temperature | `0.2` |
| Max tokens argument | `400` |
| Inter-case delay | `70s` |
| Successful cases | `10 / 10` |

Official Groq Free Plan limits for `llama-3.1-8b-instant` are listed as:
30 RPM, 14.4K RPD, 6K TPM, and 500K TPD. Source:
<https://console.groq.com/docs/rate-limits>.

The 70-second cooldown is conservative because the case prompts are large and
the free-plan TPM limit is the tightest practical constraint.

---

## 3. Inputs Used By Each System

### Metrics-Only Baseline

The metrics-only baseline reads `results/cases.json` and uses deterministic
fields already computed by the pipeline:

1. `heuristic_summary.flagged_issues_union`
2. per-counterfactual `heuristic_metrics.flagged_issues`
3. aggregate quality metrics such as sparsity, proximity, validity, and diversity
4. constraint warnings, kept separate from scored issue labels

It does not reason semantically. It mostly copies and normalizes deterministic
heuristic evidence into the Judge-style verdict schema.

### Single LLM

The single-LLM system also reads `results/cases.json`, but receives a compact
prompt containing:

1. original individual profile
2. prediction and confidence
3. model metadata
4. aggregate metrics
5. counterfactual changes and confidence scores
6. deterministic heuristic flags and evidence
7. issue taxonomy descriptions

It does not receive the ground-truth labels. Its job is to interpret the same
evidence and produce the same structured verdict schema as the multi-agent
Judge.

---

## 4. Artifacts Compared

| System | Result artifact |
|---|---|
| Metrics-only | `results/metrics_only_outputs/metrics_only_20260508_150327/metrics_only_results.json` |
| Single LLM | `results/debate_outputs/llama-3.1-8b-instant/single_llm_20260508_191646/single_llm_results.json` |

The Groq single-LLM run estimated total cost at about `$0.004053` for the
10-case batch.

---

## 5. Summary Scores

| Metric | Metrics-only | Single LLM | Interpretation |
|---|---:|---:|---|
| Successful cases | 10 / 10 | 10 / 10 | Both completed on the final clean run. |
| Ground-truth issue labels | 27 | 27 | Same reference labels. |
| Caught issue labels | 24 | 22 | Metrics-only caught two more labels. |
| Issue recall | 88.89% | 81.48% | Single LLM missed more draft reference issues. |
| Issue precision | 96.00% | 95.65% | Both made one extra issue prediction. |
| Issue F1 | 92.31% | 88.00% | Metrics-only is stronger on label coverage today. |
| Exact case match | 60.00% | 60.00% | Both exactly matched 6 of 10 cases. |
| Extra issue count | 1 | 1 | Both over-flagged case 4 with `extreme_working_hours`. |

Current scoring evaluates issue-label agreement. It does not yet score whether
`overall_assessment`, `severity`, or `recommended_action` match the draft human
judgment.

---

## 6. Per-Case Comparison

| Case | Ground truth | Metrics-only output | Single-LLM output | Readout |
|---:|---|---|---|---|
| 0 | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes`, `unactionable_capital_shift` | `implausible_time_dependent_change`, `too_many_changes`, `unactionable_capital_shift` | Single LLM missed both `fragile_counterfactual` and the semantic work-profile issue. |
| 1 | `fragile_counterfactual`, `implausible_time_dependent_change` | exact match | exact match | Both systems agree with the draft labels. |
| 2 | `fragile_counterfactual`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `fragile_counterfactual`, `too_many_changes`, `unactionable_capital_shift` | `fragile_counterfactual`, `too_many_changes`, `unactionable_capital_shift` | Both miss `inconsistent_work_profile`. |
| 3 | `unactionable_capital_shift` | exact match | exact match | Both catch the capital-shift problem. |
| 4 | `fragile_counterfactual`, `unactionable_capital_shift` | plus extra `extreme_working_hours` | plus extra `extreme_working_hours` | Both over-flag the hours issue. |
| 5 | `fragile_counterfactual`, `implausible_time_dependent_change` | exact match | exact match | Both systems agree with the draft labels. |
| 6 | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | misses `inconsistent_work_profile` | misses `implausible_time_dependent_change` and `inconsistent_work_profile` | Single LLM is weaker here. |
| 7 | `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes` | exact match | exact match | Both systems agree with the draft labels. |
| 8 | `fragile_counterfactual`, `implausible_time_dependent_change` | exact match | exact match | Both systems agree with the draft labels. |
| 9 | `fragile_counterfactual`, `implausible_time_dependent_change` | exact match | exact match | Both systems agree with the draft labels. |

---

## 7. What This Shows

The single LLM is operational and cleanly comparable, but it does not yet prove
added value over the deterministic baseline.

Main observations:

1. Both systems exactly match the same 6 cases: 1, 3, 5, 7, 8, and 9.
2. Both systems struggle with `inconsistent_work_profile` in cases 0, 2, and 6.
3. The single LLM additionally misses `fragile_counterfactual` in case 0 and
   `implausible_time_dependent_change` in case 6.
4. Both systems add the same extra issue in case 4: `extreme_working_hours`.
5. The single LLM gives richer natural-language rationales, but its issue-label
   performance is currently lower than metrics-only.

The important result is negative but useful: a single general LLM, even with the
same evidence and taxonomy, did not automatically solve the semantic gap. That
means the later multi-agent stage has a clear target.

---

## 8. Scientific Interpretation

At this point, the metrics-only baseline remains the stronger issue-label
baseline. The single LLM is still valuable because it establishes the middle
condition between deterministic scoring and adversarial debate:

```text
metrics-only baseline
  -> deterministic threshold/heuristic reading

single LLM
  -> one semantic judge reading the same evidence

multi-agent system
  -> adversarial semantic review before final judgment
```

For the multi-agent system to demonstrate added value, it should ideally:

1. keep the high precision of metrics-only;
2. recover semantic labels such as `inconsistent_work_profile`;
3. avoid adding unsupported issues like case 4's `extreme_working_hours`;
4. produce reasoning that is more auditable than the single-LLM response;
5. match or exceed metrics-only issue recall and exact-match rate.

The next experiment should therefore focus on whether adversarial debate helps
with the labels the current systems miss, especially `inconsistent_work_profile`.
