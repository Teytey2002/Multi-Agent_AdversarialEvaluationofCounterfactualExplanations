# Multi-Agent Structured Protocol Results

Date: 2026-05-15  
Final run artifact: `results/debate_outputs/llama-3.1-8b-instant/multi_agent_20260515_193246/multi_agent_results.json`  
Transcript directory: `results/debate_outputs/llama-3.1-8b-instant/multi_agent_20260515_193246/transcripts/`  
Reference labels: `annotations/ground_truth_labels.json`

This report records the multi-agent benchmark after tightening the debate
protocol to avoid truncated free-form specialist answers.

---

## 1. Fixes Applied Before The Final Run

The previous multi-agent runs used a `250` completion-token budget for every
agent. This was enough for the Judge JSON, but the specialist turns were often
cut mid-sentence. Increasing the token budget directly would also increase the
later Judge prompt size, so the implemented fix was to reduce specialist
verbosity instead.

Changes:

1. Added a compact four-line specialist protocol:
   `ISSUES_SUPPORTED_BY_EVIDENCE`, `ISSUES_NOT_SUPPORTED_OR_OVERSTATED`,
   `KEY_EVIDENCE`, and `BOTTOM_LINE`.
2. Added explicit instructions not to copy taxonomy definitions or list every
   allowed label.
3. Clarified that only Prosecutor, Defense, and Expert Witness use the compact
   specialist format; Judge must return JSON only.
4. Hardened verdict parsing so the runner uses the latest parseable Judge JSON
   instead of blindly parsing the final Judge message.
5. Removed the exact `VERDICT_COMPLETE` sentinel from the task prompt because
   it could prematurely trigger AutoGen's text termination condition.
6. Aligned the actual prompt taxonomy with Theo's narrowed
   `inconsistent_work_profile` policy: this label should only be used when
   deterministic heuristic evidence reports a direct workclass/occupation
   contradiction.
7. Clarified that `generation_policy.permitted_range` is a DiCE generation
   bound, not an actionability guarantee. A capital shift can be inside the
   permitted range and still be unactionable if heuristic evidence says the
   jump is too large.

Validation before and after the final run:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Result: 21 tests passed; compileall passed.

---

## 2. Final Run Configuration

The final benchmark reused the same quota-safe settings as the May 9 and May 15
comparison runs:

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

All 10 cases completed successfully.

---

## 3. Transcript Sanity Check

The final transcript audit found:

| Check | Result |
|---|---|
| Successful cases | 10 / 10 |
| Runtime failures | 0 |
| Transcripts written | 10 / 10 |
| Judge JSON present | 10 / 10 |
| Empty Judge messages | 0 |
| Specialist section format present | 10 / 10 |
| Specialist completions hitting 250-token cap | 0 |

This fixes the earlier transcript-quality problem. The specialist turns are now
complete and compact enough for the Judge to receive a full debate context.

---

## 4. Headline Metrics

| Metric | May 9 baseline | May 15 Theo-only | May 15 structured final |
|---|---:|---:|---:|
| Successful cases | 10 / 10 | 10 / 10 | 10 / 10 |
| Predicted labels | 33 | 22 | 24 |
| True positives | 18 | 14 | 19 |
| False positives | 15 | 8 | 5 |
| False negatives | 9 | 13 | 8 |
| Precision | 54.55% | 63.64% | 79.17% |
| Recall | 66.67% | 51.85% | 70.37% |
| F1 | 60.00% | 57.14% | 74.51% |
| Exact case match | 0.00% | 10.00% | 20.00% |

The structured protocol improves over both previous multi-agent runs on
precision, recall, F1, and exact-match rate.

---

## 5. Per-Case Final Results

| Case | Ground truth | Final output | Readout |
|---:|---|---|---|
| 0 | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `implausible_time_dependent_change`, `too_many_changes`, `unactionable_capital_shift` | Misses `fragile_counterfactual` and `inconsistent_work_profile`. |
| 1 | `fragile_counterfactual`, `implausible_time_dependent_change` | exact match | Improvement. |
| 2 | `fragile_counterfactual`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `fragile_counterfactual`, `too_many_changes`, `unactionable_capital_shift` | Misses `inconsistent_work_profile`. |
| 3 | `unactionable_capital_shift` | plus extra `fragile_counterfactual` | Mostly correct but over-flags fragility. |
| 4 | `fragile_counterfactual`, `unactionable_capital_shift` | plus extra `extreme_working_hours` | Mostly correct but over-flags hours. |
| 5 | `fragile_counterfactual`, `implausible_time_dependent_change` | exact match | Improvement. |
| 6 | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `implausible_time_dependent_change`, `unactionable_capital_shift` | Misses `inconsistent_work_profile` and `too_many_changes`. |
| 7 | `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes` | `implausible_time_dependent_change`, `too_many_changes` | Misses `fragile_counterfactual`. |
| 8 | `fragile_counterfactual`, `implausible_time_dependent_change` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes` | Misses fragility and adds two unsupported issues. |
| 9 | `fragile_counterfactual`, `implausible_time_dependent_change` | `implausible_time_dependent_change`, `inconsistent_work_profile` | Misses fragility and adds unsupported work-profile issue. |

---

## 6. Label-Level Behavior

| Label | May 9 TP/FP/FN | Theo-only TP/FP/FN | Structured final TP/FP/FN | Readout |
|---|---|---|---|---|
| `extreme_working_hours` | 0 / 2 / 0 | 0 / 1 / 0 | 0 / 1 / 0 | Still one unsupported extra. |
| `fragile_counterfactual` | 2 / 0 / 6 | 2 / 0 / 6 | 4 / 1 / 4 | Improved recall but still under-detected. |
| `implausible_time_dependent_change` | 7 / 2 / 0 | 5 / 1 / 2 | 7 / 0 / 0 | Solved in this run. |
| `inconsistent_work_profile` | 3 / 6 / 0 | 2 / 3 / 1 | 0 / 2 / 3 | Over-corrected against this draft label. |
| `too_many_changes` | 2 / 4 / 2 | 1 / 3 / 3 | 3 / 1 / 1 | Improved substantially. |
| `unactionable_capital_shift` | 4 / 1 / 1 | 4 / 0 / 1 | 5 / 0 / 0 | Solved in this run. |

---

## 7. Interpretation

The structured protocol fixed the mechanical transcript problem. The May 9 and
Theo-only runs were hard to interpret because specialist turns could be cut
while the Judge still produced a JSON verdict. The final run no longer has that
problem: every specialist response is complete, and the Judge receives a
complete compact debate.

Behaviorally, the final run is also the strongest multi-agent run so far:

```text
May 9 baseline F1:        60.00%
May 15 Theo-only F1:      57.14%
Structured final F1:      74.51%
```

The biggest improvements are:

1. fewer unsupported labels overall;
2. perfect detection of `implausible_time_dependent_change`;
3. perfect detection of `unactionable_capital_shift`;
4. better handling of `too_many_changes`;
5. no runtime failures and no cut specialist turns.

The main remaining problem is `inconsistent_work_profile`. The actual prompt
policy now says this label requires deterministic heuristic evidence, but the
draft ground truth still contains it in cases where the deterministic heuristic
layer does not support it. As a result, the final run misses all three draft
ground-truth instances while still adding two unsupported instances.

This may not only be an LLM problem. It may indicate a mismatch between the
draft human labels and the narrowed deterministic taxonomy policy.

---

## 8. Next Logical Step

The next step should be a label-policy review, not another debate run.

Specifically:

1. Decide whether `inconsistent_work_profile` is supposed to be a deterministic
   heuristic label only, or a broader semantic human judgment label.
2. If it is deterministic only, update the draft ground-truth annotations so
   they do not expect unsupported work-profile labels.
3. If it is broader semantic judgment, then the heuristic layer must be expanded
   to produce explicit work-profile evidence for cases 0, 2, and 6.

Until that choice is made, the multi-agent system is being asked to satisfy two
different definitions of the same label.

