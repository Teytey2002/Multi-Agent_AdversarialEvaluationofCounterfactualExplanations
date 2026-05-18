# Multi-Agent Prompt Calibration Comparison

Date: 2026-05-15  
Baseline artifact: `results/debate_outputs/llama-3.1-8b-instant/multi_agent_20260509_091856/multi_agent_results.json`  
New artifact: `results/debate_outputs/llama-3.1-8b-instant/multi_agent_20260515_163553/multi_agent_results.json`  
Reference labels: `annotations/ground_truth_labels.json`  

This report compares the May 9 multi-agent run against a new controlled run
after commit `14f965e` by `teytey2002`:

```text
14f965e update for stop hallucinate
```

The comparison uses the same 10 cases, the same draft ground-truth labels, the
same model, and the same quota-safe multi-agent configuration. The purpose is
to isolate whether the anti-hallucination prompt changes improved or regressed
the multi-agent evaluation behavior.

---

## 1. Controlled Run Configuration

The new run used the same configuration documented in
`docs/reports/multi_agent_final_results_2026-05-09.md`:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\run_debate.py --max-rounds 1 --turn-delay 70 --delay 70 --max-tokens 250
```

| Setting | Value |
|---|---:|
| Provider | `groq` |
| Model | `llama-3.1-8b-instant` |
| Temperature | `0.2` |
| Max tokens | `250` |
| Speaker selection | `round_robin` |
| Specialist rounds | `1` |
| Turn delay | `70s` |
| Inter-case delay | `70s` |
| Cases | `0-9` |

The run completed successfully for all 10 cases.

---

## 2. What Changed In Theo's Commit

The latest commit by `teytey2002` did not change the case data, ground-truth
labels, or the evaluation script. It changed the multi-agent prompt behavior:

1. Prosecutor must identify only issues supported by explicit heuristic evidence.
2. Defense must challenge labels that lack deterministic support.
3. Expert Witness must not invent plausibility concerns from general reasoning.
4. Judge must only flag issue labels when deterministic evidence supports them.
5. `inconsistent_work_profile` was narrowed to direct deterministic
   workclass/occupation contradictions.
6. Prompt payloads were compacted further with minified JSON.

The intended effect was to reduce unsupported semantic over-flagging.

---

## 3. Headline Metrics

| Metric | May 9 baseline | May 15 after `teytey2002` | Change |
|---|---:|---:|---:|
| Successful cases | 10 / 10 | 10 / 10 | same |
| Total ground-truth labels | 27 | 27 | same |
| Predicted labels | 33 | 22 | -11 |
| True positives | 18 | 14 | -4 |
| False positives | 15 | 8 | -7 |
| False negatives | 9 | 13 | +4 |
| Precision | 54.55% | 63.64% | +9.09 pp |
| Recall | 66.67% | 51.85% | -14.82 pp |
| F1 | 60.00% | 57.14% | -2.86 pp |
| Exact case match | 0.00% | 10.00% | +10.00 pp |

Important note: the runner's printed `false_positive_rate` is not a label-level
false-positive rate. It measures false-positive clean cases, and all 10 cases in
this benchmark have at least one ground-truth issue. The more useful number here
is label-level false positives: 15 before, 8 after.

---

## 4. Per-Case Comparison

| Case | Ground truth | May 9 output | May 15 output | Readout |
|---:|---|---|---|---|
| 0 | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `unactionable_capital_shift` | `extreme_working_hours`, `inconsistent_work_profile`, `unactionable_capital_shift` | Regression: lost `implausible_time_dependent_change` and added unsupported `extreme_working_hours`. |
| 1 | `fragile_counterfactual`, `implausible_time_dependent_change` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes` | exact match | Improvement: removed two extras and recovered `fragile_counterfactual`. |
| 2 | `fragile_counterfactual`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | all GT labels plus extra `implausible_time_dependent_change` | `inconsistent_work_profile` only | Regression: over-corrected and missed three GT labels. |
| 3 | `unactionable_capital_shift` | `inconsistent_work_profile`, `unactionable_capital_shift` | `too_many_changes`, `unactionable_capital_shift` | Mixed: removed wrong work-profile flag but added wrong burden flag. |
| 4 | `fragile_counterfactual`, `unactionable_capital_shift` | `extreme_working_hours`, `implausible_time_dependent_change`, `inconsistent_work_profile` | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `unactionable_capital_shift` | Improvement on recall, but still two extras. |
| 5 | `fragile_counterfactual`, `implausible_time_dependent_change` | all GT labels plus three extras | `implausible_time_dependent_change` only | Mixed: removed extras but now misses `fragile_counterfactual`. |
| 6 | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | missed `too_many_changes` | missed `inconsistent_work_profile` | Roughly neutral: still misses one label, but a different one. |
| 7 | `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes` | missed `fragile_counterfactual`, added `inconsistent_work_profile` | `inconsistent_work_profile` only | Regression: now misses all GT labels and keeps the extra. |
| 8 | `fragile_counterfactual`, `implausible_time_dependent_change` | missed `fragile_counterfactual`, added two extras | missed `fragile_counterfactual`, added `too_many_changes` | Slight improvement: fewer extras, same missed fragility. |
| 9 | `fragile_counterfactual`, `implausible_time_dependent_change` | missed `fragile_counterfactual`, added two extras | same as May 9 | No change. |

---

## 5. Label-Level Behavior

| Label | GT count | May 9 predicted | May 15 predicted | May 9 TP/FP/FN | May 15 TP/FP/FN | Readout |
|---|---:|---:|---:|---|---|---|
| `extreme_working_hours` | 0 | 2 | 1 | 0 / 2 / 0 | 0 / 1 / 0 | Improved, but still one unsupported flag. |
| `fragile_counterfactual` | 8 | 2 | 2 | 2 / 0 / 6 | 2 / 0 / 6 | No improvement; still badly under-detected. |
| `implausible_time_dependent_change` | 7 | 9 | 6 | 7 / 2 / 0 | 5 / 1 / 2 | Fewer extras, but now misses true cases. |
| `inconsistent_work_profile` | 3 | 9 | 5 | 3 / 6 / 0 | 2 / 3 / 1 | Major precision improvement, but not solved. |
| `too_many_changes` | 4 | 6 | 4 | 2 / 4 / 2 | 1 / 3 / 3 | Slightly fewer extras, worse recall. |
| `unactionable_capital_shift` | 5 | 5 | 4 | 4 / 1 / 1 | 4 / 0 / 1 | Improved precision with same recall. |

---

## 6. Interpretation

Theo's anti-hallucination changes worked in the narrow sense: the multi-agent
system became less eager to invent issues. Total predicted labels dropped from
33 to 22, and label-level false positives dropped from 15 to 8.

The clearest improvement is `inconsistent_work_profile`. The May 9 system
flagged it in 9 of 10 cases even though the draft labels contain it in only 3
cases. The May 15 system flags it in 5 of 10 cases. That is still too high, but
it is a meaningful reduction in over-generalization.

However, the change also over-corrected. True positives dropped from 18 to 14,
false negatives rose from 9 to 13, and recall dropped from 66.67% to 51.85%.
The system is now more conservative, but also misses more actual ground-truth
issues.

The F1 score moved from 60.00% to 57.14%, so the net label-level result is a
small regression despite better precision. Exact case match improved from 0 to
1 case because case 1 is now exactly correct.

---

## 7. Conclusion

The May 15 run shows a calibration tradeoff:

```text
May 9 behavior:
  higher recall, too many unsupported semantic labels

May 15 behavior after teytey2002:
  better precision, fewer hallucinated labels, but too much under-detection
```

This is not a full improvement yet. It is a useful intermediate correction:
Theo's prompt changes reduced the main May 9 failure mode, but the system now
needs a second calibration pass focused on recovering missed deterministic
issues, especially `fragile_counterfactual`, `too_many_changes`, and valid
`implausible_time_dependent_change` cases.

The practical next step is not to add debate rounds. The next step is to make
the Judge copy or strongly respect the deterministic `heuristic_metrics`
evidence for issue labels, while still using the agents' debate only to explain
and contextualize those labels.

