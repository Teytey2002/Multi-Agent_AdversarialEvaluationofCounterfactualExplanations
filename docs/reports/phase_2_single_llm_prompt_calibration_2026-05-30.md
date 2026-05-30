# Phase 2 Single-LLM Prompt Calibration Results

Date: 2026-05-30  
Dataset artifact: `results/cases.json`  
Reference system: `results/metrics_only_outputs/metrics_only_latest.json`  
Model: `llama-3.1-8b-instant` via Groq  
Mode: single-LLM only, no multi-agent rerun

---

## 1. Purpose

This report documents one focused Phase 2 calibration pass for the single-LLM
evaluator.

Phase 2 reframes the evaluation as a substitution-feasibility study. The
metrics-only baseline is treated as the deterministic reference system. The
single LLM is therefore evaluated by how closely it can approximate that
reference, not by whether it discovers an independent external truth.

The uncalibrated Phase 2 run was already informative. It showed that the
single LLM could follow the active five-label taxonomy reasonably well, but it
also exposed a systematic failure pattern: the model often missed
`fragile_counterfactual`, even when the deterministic heuristic layer had
already flagged it.

The calibration objective was deliberately narrow:

1. make the single LLM more faithful to deterministic heuristic evidence;
2. reduce silent dropping of candidate issue labels;
3. especially preserve `fragile_counterfactual` when it is present in case-level
   or CF-level heuristic flags;
4. avoid repeated prompt tuning or score chasing.

---

## 2. Prompt Calibration

The single evaluator system message was updated in `src/agents/agents.py`.

Conceptually, the change adds a Phase 2 substitution frame:

- the single LLM should approximate the metrics-only reference system;
- it should not invent a new evaluation policy;
- deterministic heuristic evidence is the primary issue-label source;
- `heuristic_summary.flagged_issues_union` is the starting candidate set;
- candidate issues should be retained unless the case evidence internally
  contradicts them;
- a label absent from both case-level and CF-level heuristic flags should not be
  added;
- `fragile_counterfactual` must not be dropped merely because other issues are
  more severe.

The taxonomy itself was not changed. The calibration changes the evaluator's
use of the existing evidence, not the underlying issue definitions.

A regression test was added to verify that the single-evaluator system prompt
contains the Phase 2 substitution framing, the heuristic-union rule, and the
explicit `fragile_counterfactual` preservation rule.

---

## 3. Experimental Setup

The calibrated single-LLM run used the same practical configuration as the
uncalibrated run:

```powershell
$env:PYTHONPATH="src"; python scripts/run_debate.py --single-llm --max-tokens 400
```

Run artifacts:

| Run | Artifact |
|---|---|
| Uncalibrated Phase 2 single LLM | `results/debate_outputs/llama-3.1-8b-instant/single_llm_20260530_153142/single_llm_results.json` |
| Calibrated Phase 2 single LLM | `results/debate_outputs/llama-3.1-8b-instant/single_llm_20260530_164601/single_llm_results.json` |
| Calibrated latest copy | `results/debate_outputs/llama-3.1-8b-instant_single_llm_latest.json` |

The substitution scorer was then rerun:

```powershell
$env:PYTHONPATH="src"; python scripts/score_against_baseline.py
```

Substitution artifacts:

| Run | Artifact |
|---|---|
| Uncalibrated substitution scores | `results/substitution_outputs/substitution_scores_20260530_154312.json` |
| Calibrated substitution scores | `results/substitution_outputs/substitution_scores_20260530_165706.json` |
| Calibrated latest copy | `results/substitution_outputs/substitution_scores_latest.json` |

No multi-agent system run was performed.

---

## 4. Substitution Results

| Metric | Uncalibrated single LLM | Calibrated single LLM | Change |
|---|---:|---:|---:|
| Detection rate | 84.0% | 96.0% | +12.0 pp |
| False-positive rate | 0.0% | 0.0% | 0.0 pp |
| Exact-match rate | 60.0% | 90.0% | +30.0 pp |
| Assessment agreement | 50.0% | 50.0% | 0.0 pp |
| Severity agreement | 60.0% | 50.0% | -10.0 pp |
| Recommended-action agreement | 70.0% | 60.0% | -10.0 pp |
| Perfect issue-set matches | 6 / 10 | 9 / 10 | +3 cases |

The calibration substantially improved issue-label substitution. It eliminated
three of the four previous issue-set disagreements without introducing extra
issue labels.

The improvement is concentrated exactly where expected: the model became more
faithful to heuristic-supported labels, especially `fragile_counterfactual`.

---

## 5. Case-Level Disagreement Analysis

Before calibration, the single LLM disagreed with the metrics-only reference on
four cases:

| Case | Uncalibrated missed issues | Uncalibrated extra issues |
|---:|---|---|
| 0 | `fragile_counterfactual` | none |
| 2 | `fragile_counterfactual` | none |
| 6 | `implausible_time_dependent_change` | none |
| 7 | `fragile_counterfactual` | none |

After calibration, only one issue-set disagreement remains:

| Case | Calibrated missed issues | Calibrated extra issues |
|---:|---|---|
| 0 | `fragile_counterfactual` | none |

The calibrated model recovered:

- `fragile_counterfactual` on case 2;
- `implausible_time_dependent_change` on case 6;
- `fragile_counterfactual` on case 7.

The remaining case 0 disagreement is useful rather than merely negative. It
shows that even explicit prompt calibration does not guarantee perfect
substitution. The LLM still sometimes prioritizes the most semantically salient
issues and omits a lower-severity robustness label.

That residual failure supports the project's case-level analysis requirement:
disagreements should be inspected, not automatically treated as LLM failure or
reference-system infallibility.

---

## 6. Interpretation

The calibrated prompt improves substitution feasibility for issue labels. The
single LLM now reproduces 24 of 25 reference issue labels, with no extra labels.
This is strong evidence that a small single LLM can approximate a structured
heuristic reference when the prompt explicitly frames the task as reference
substitution rather than open-ended evaluation.

However, the scalar verdict fields did not improve. Assessment agreement stayed
at 50%, while severity and recommended-action agreement decreased slightly. This
reveals a second limitation: matching the issue set is easier than matching the
reference system's full verdict policy.

The likely reason is that the single LLM tends to use a harsher qualitative
judgment once any issue is present. The metrics-only reference distinguishes
more carefully between ambiguous/medium cases and unfair/high cases. The prompt
calibration targeted issue-label substitution, not full severity calibration.

Therefore the academically defensible conclusion is nuanced:

> Focused prompt calibration substantially improves issue-label substitution,
> but the 8B single LLM still does not fully substitute the deterministic
> reference on overall assessment, severity, and recommended action.

---

## 7. Legacy Annotation-Score Context

The script `run_debate.py` still prints scores against the legacy labels embedded
in `results/cases.json`. These scores are useful as historical context but are
not the main Phase 2 evaluation target, because those labels still include the
removed `inconsistent_work_profile` issue.

For completeness:

| Metric | Uncalibrated | Calibrated |
|---|---:|---:|
| Legacy-label detection | 74.07% | 85.19% |
| Legacy-label exact match | 50.0% | 60.0% |
| Successful cases | 10 / 10 | 10 / 10 |

The legacy-label improvement is directionally consistent with the substitution
result, but the substitution score remains the cleaner Phase 2 measurement.

---

## 8. Conclusion

This calibration pass should be treated as the final single-LLM prompt
calibration for the project unless a new research question is introduced.

It establishes three points:

1. The uncalibrated Phase 2 single LLM was a valid baseline, not wasted work.
2. A narrow, evidence-faithful prompt improved issue-label substitution from
   84% to 96% detection and from 60% to 90% exact match.
3. Full verdict substitution remains incomplete because scalar assessment,
   severity, and action decisions are not yet aligned with the metrics-only
   reference policy.

This is enough for the final report: the project can now argue that LLMs are
promising substitutes for structured issue identification, but not yet reliable
standalone replacements for the complete deterministic evaluation policy.
