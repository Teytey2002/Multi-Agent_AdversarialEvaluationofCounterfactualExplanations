# Multi-Agent Evaluation Results

Date: 2026-05-09  
Dataset artifact: `results/cases.json`  
Reference labels: `annotations/ground_truth_labels.json`

This report records the first complete multi-agent run after the pipeline,
ground-truth, metrics-only, single-LLM, and repository-structure updates.

---

## 1. Readiness Check

The multi-agent system is up to date with the recent project changes:

| Area | Status |
|---|---|
| Reorganized imports | Passes with the new `src/pipeline/`, `src/policy/`, and `scripts/` layout. |
| Ground-truth leakage | `ground_truth_issues`, `ground_truth_by_cf`, and `ground_truth_source` are excluded from the debate prompt. |
| Shared schema | Multi-agent Judge returns the same verdict schema as metrics-only and single-LLM. |
| Groq-only config | Still uses the Groq OpenAI-compatible endpoint through AutoGen. |
| Developer manageability | Run uses deterministic `round_robin`, one specialist round, and saved transcripts per case. |

Validation before running:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m unittest discover -s tests
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m compileall -q src scripts tests
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\run_debate.py --help
```

Result: 18 tests passed, compileall passed, and CLI help loaded correctly.

---

## 2. Quota-Safe Run Strategy

Official Groq Free Plan limits for `llama-3.1-8b-instant` are 30 RPM,
14.4K RPD, 6K TPM, and 500K TPD. Source:
<https://console.groq.com/docs/rate-limits>.

The limiting factor is TPM. A multi-agent debate has multiple LLM calls per
case, and each later turn includes more conversation history. The default
two-round debate would be too large for the free tier.

The chosen run configuration was therefore:

```powershell
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\run_debate.py --max-rounds 1 --turn-delay 70 --delay 70 --max-tokens 250
```

| Setting | Value | Reason |
|---|---:|---|
| `--max-rounds` | `1` | Keeps the debate interpretable: Prosecutor, Defense, Expert Witness, then Judge. |
| `--turn-delay` | `70s` | Paces specialist turns so requests do not stack inside the TPM window. |
| `--delay` | `70s` | Paces cases. |
| `--max-tokens` | `250` | Keeps the largest Judge request under the 6K token limit. |
| `--speaker-selection` | `round_robin` | Deterministic and easier to audit than auto-selection. |

An initial smoke test with `--max-tokens 350` failed on case 0 because the
Judge request asked for 6005 tokens against a 6000-token limit. Reducing to
`250` completed successfully.

---

## 3. Run Artifact

Final full-run artifact:

```text
results/debate_outputs/llama-3.1-8b-instant/multi_agent_20260509_091856/multi_agent_results.json
```

Transcripts:

```text
results/debate_outputs/llama-3.1-8b-instant/multi_agent_20260509_091856/transcripts/
```

All 10 cases completed successfully.

Estimated total run cost: about `$0.011197`.

---

## 4. Headline Scores

| System | Precision | Recall | F1 | Exact Match | Successful Cases |
|---|---:|---:|---:|---:|---:|
| Metrics-only | 96.00% | 88.89% | 92.31% | 60.00% | 10 / 10 |
| Single LLM | 95.65% | 81.48% | 88.00% | 60.00% | 10 / 10 |
| Multi-agent | 54.55% | 66.67% | 60.00% | 0.00% | 10 / 10 |

The multi-agent run did not outperform the baselines. It completed reliably,
but it over-flagged many labels and exactly matched none of the 10 cases.

---

## 5. Per-Case Results

| Case | Ground Truth | Multi-Agent Output | Result |
|---:|---|---|---|
| 0 | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `unactionable_capital_shift` | Missed `fragile_counterfactual`, `too_many_changes`. |
| 1 | `fragile_counterfactual`, `implausible_time_dependent_change` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes` | Missed `fragile_counterfactual`; added two issues. |
| 2 | `fragile_counterfactual`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `fragile_counterfactual`, `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | Added `implausible_time_dependent_change`. |
| 3 | `unactionable_capital_shift` | `inconsistent_work_profile`, `unactionable_capital_shift` | Added `inconsistent_work_profile`. |
| 4 | `fragile_counterfactual`, `unactionable_capital_shift` | `extreme_working_hours`, `implausible_time_dependent_change`, `inconsistent_work_profile` | Missed both reference labels; added three issues. |
| 5 | `fragile_counterfactual`, `implausible_time_dependent_change` | `extreme_working_hours`, `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes`, `unactionable_capital_shift` | Caught both reference labels, but added three issues. |
| 6 | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes`, `unactionable_capital_shift` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `unactionable_capital_shift` | Missed `too_many_changes`. |
| 7 | `fragile_counterfactual`, `implausible_time_dependent_change`, `too_many_changes` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes` | Missed `fragile_counterfactual`; added `inconsistent_work_profile`. |
| 8 | `fragile_counterfactual`, `implausible_time_dependent_change` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes` | Missed `fragile_counterfactual`; added two issues. |
| 9 | `fragile_counterfactual`, `implausible_time_dependent_change` | `implausible_time_dependent_change`, `inconsistent_work_profile`, `too_many_changes` | Missed `fragile_counterfactual`; added two issues. |

---

## 6. Interpretation

The multi-agent system shows one useful behavior: it recovered
`inconsistent_work_profile` in the three cases where the draft labels include
that semantic issue: cases 0, 2, and 6.

However, it overgeneralized that same label heavily. It predicted
`inconsistent_work_profile` in 9 of 10 cases, even though the reference labels
contain it in only 3 cases. This explains the low precision.

Main failure modes:

1. **Over-flagging semantic labels**: especially `inconsistent_work_profile`.
2. **Under-detecting fragility**: it missed `fragile_counterfactual` in several cases where both baselines often caught it.
3. **More severe bias than single LLM**: the adversarial framing appears to push the Judge toward rejection and extra issues.
4. **Quota-constrained compression**: one round and `250` max tokens keep the run feasible, but may reduce nuance.

---

## 7. Scientific Conclusion

Current ranking by issue-label agreement:

```text
1. Metrics-only baseline
2. Single LLM
3. Multi-agent, one-round Groq-free-tier configuration
```

This is a negative but important result. The multi-agent mechanism is runnable
and auditable, but it does not yet demonstrate added value. The next improvement
should focus on calibration, not more debate volume:

1. make the Judge require direct evidence for every flagged issue;
2. make Defense explicitly challenge unsupported issue labels;
3. add a final "label calibration" instruction before the Judge verdict;
4. keep `round_robin` and one round until precision improves;
5. retest against the same 10 cases after prompt calibration.

The multi-agent system should not be presented as better than the baselines in
its current form. Its value right now is diagnostic: it reveals that adversarial
pressure can increase semantic sensitivity, but also increases false positives.
