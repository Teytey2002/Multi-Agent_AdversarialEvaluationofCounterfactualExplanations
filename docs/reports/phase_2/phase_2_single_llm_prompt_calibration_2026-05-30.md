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

Implementation evidence: the actual calibration block added in
`src/agents/agents.py` is shown below. This is the central behavioral change;
it makes the LLM start from the deterministic heuristic union and explicitly
preserve `fragile_counterfactual` when the evidence supports it.

```python
SINGLE_EVALUATOR_PHASE2_CALIBRATION = """
Phase 2 substitution calibration:
- Your task is to approximate the metrics-only reference system, not to invent a new evaluation policy.
- Treat deterministic heuristic evidence as the primary issue-label source.
- Start from `heuristic_summary.flagged_issues_union` as candidate scored issues.
- Include every candidate issue unless the case evidence internally contradicts it.
- Do not add a scored label absent from both `heuristic_summary.flagged_issues_union` and all `counterfactuals[*].heuristic_metrics.flagged_issues`.
- If `fragile_counterfactual` appears in case-level or CF-level heuristic flags, include it in `flagged_issues`.
- Treat `cf_confidence` near the 0.5 decision boundary as prediction-robustness evidence.
- Do not drop `fragile_counterfactual` because other issues are more severe.
""".strip()
```

A regression test was added to verify that the single-evaluator system prompt
contains the Phase 2 substitution framing, the heuristic-union rule, and the
explicit `fragile_counterfactual` preservation rule.

```python
def test_single_evaluator_has_phase_2_substitution_calibration(self):
    system_message = _build_single_evaluator_system_message()

    self.assertIn("Phase 2 substitution calibration", system_message)
    self.assertIn("metrics-only reference system", system_message)
    self.assertIn("heuristic_summary.flagged_issues_union", system_message)
    self.assertIn("Include every candidate issue", system_message)
    self.assertIn("absent from both", system_message)
    self.assertIn("fragile_counterfactual", system_message)
    self.assertIn("Do not drop `fragile_counterfactual`", system_message)
    self.assertIn("cf_confidence", system_message)
```

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

The calibrated single-LLM command completed the full 10-case sample and wrote
the latest single-LLM artifact. The relevant execution evidence is compactly:

```text
Mode: single_llm
Model: llama-3.1-8b-instant
Successful cases: 10 / 10
Latest output: results/debate_outputs/llama-3.1-8b-instant_single_llm_latest.json
```

---

## 4. Substitution Results

Metrics use the unified Phase 1 vocabulary (precision / recall / F1), with each
reference issue label treated as the positive class. `recall` is the measure
previously reported as `detection_rate`.

| Metric | Uncalibrated single LLM | Calibrated single LLM | Change |
|---|---:|---:|---:|
| Precision | 100.0% | 100.0% | 0.0 pp |
| Recall | 84.0% | 96.0% | +12.0 pp |
| F1 | 91.3% | 98.0% | +6.7 pp |
| Exact-match rate | 60.0% | 90.0% | +30.0 pp |
| Assessment agreement | 50.0% | 50.0% | 0.0 pp |
| Severity agreement | 60.0% | 50.0% | -10.0 pp |
| Recommended-action agreement | 70.0% | 60.0% | -10.0 pp |
| Perfect issue-set matches | 6 / 10 | 9 / 10 | +3 cases |

Both runs had zero false-positive labels, so precision is 100% in each; the
gain is entirely in recall (and therefore F1). The same result appears directly
in the scorer output. The uncalibrated substitution summary was:

```json
{
  "precision": 100.0,
  "recall": 84.0,
  "f1": 91.3,
  "exact_match_rate": 60.0,
  "assessment_agreement": 50.0,
  "severity_agreement": 60.0,
  "recommended_action_agreement": 70.0,
  "true_positives": 21,
  "false_positives": 0,
  "false_negatives": 4,
  "total_cases": 10,
  "cases_with_perfect_match": 6
}
```

The calibrated substitution summary became:

```json
{
  "precision": 100.0,
  "recall": 96.0,
  "f1": 98.0,
  "exact_match_rate": 90.0,
  "assessment_agreement": 50.0,
  "severity_agreement": 50.0,
  "recommended_action_agreement": 60.0,
  "true_positives": 24,
  "false_positives": 0,
  "false_negatives": 1,
  "total_cases": 10,
  "cases_with_perfect_match": 9
}
```

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

The relevant case input already contained the fragility evidence. In
`results/cases.json`, case 7 exposes `fragile_counterfactual` in the
case-level heuristic union, with CF confidence values close to the 0.5 decision
boundary:

```json
{
  "case_id": 7,
  "heuristic_summary": {
    "flagged_issues_union": [
      "fragile_counterfactual",
      "implausible_time_dependent_change",
      "too_many_changes"
    ],
    "issue_evidence": {
      "fragile_counterfactual": [
        {
          "cf_confidence": 0.5484362850489894,
          "decision_threshold": 0.5,
          "fragility_threshold": 0.6,
          "margin_above_threshold": 0.04843628504898945
        },
        {
          "cf_confidence": 0.5530423109215783,
          "decision_threshold": 0.5,
          "fragility_threshold": 0.6,
          "margin_above_threshold": 0.053042310921578295
        }
      ]
    }
  }
}
```

The case-level scorer output makes the recovery visible. Before calibration,
case 7 missed the reference fragility label:

```json
{
  "case_id": 7,
  "reference_flagged_issues": [
    "fragile_counterfactual",
    "implausible_time_dependent_change",
    "too_many_changes"
  ],
  "single_llm": {
    "flagged_issues": [
      "implausible_time_dependent_change",
      "too_many_changes"
    ],
    "missed_issues": ["fragile_counterfactual"],
    "extra_issues": [],
    "exact_match": false
  }
}
```

After calibration, the same case became an exact issue-set match:

```json
{
  "case_id": 7,
  "reference_flagged_issues": [
    "fragile_counterfactual",
    "implausible_time_dependent_change",
    "too_many_changes"
  ],
  "single_llm": {
    "flagged_issues": [
      "fragile_counterfactual",
      "implausible_time_dependent_change",
      "too_many_changes"
    ],
    "missed_issues": [],
    "extra_issues": [],
    "exact_match": true
  }
}
```

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
| Legacy-label recall | 74.07% | 85.19% |
| Legacy-label exact match | 50.0% | 60.0% |
| Successful cases | 10 / 10 | 10 / 10 |

The legacy-label improvement is directionally consistent with the substitution
result, but the substitution score remains the cleaner Phase 2 measurement.

---

## 8. Conclusion

This calibration pass should be treated as the final single-LLM prompt
calibration for the project unless a new research question is introduced.

It establishes three empirical points:

1. The uncalibrated Phase 2 single LLM was a valid baseline, not wasted work.
2. A narrow, evidence-faithful prompt improved issue-label substitution from
   84% to 96% recall and from 60% to 90% exact match.
3. Full verdict substitution remains incomplete because scalar assessment,
   severity, and action decisions are not yet aligned with the metrics-only
   reference policy.

The broader implication is not that the single LLM simply "beats" or "fails to
beat" the metrics-only reference. The two systems have different strengths. The
metrics-only reference remains superior on cost, reproducibility, auditability,
and batch-scale stability. It is deterministic, free to run after implementation,
and easier to defend when the evaluation target is a fixed rule policy.

The calibrated single LLM becomes valuable on different comparison axes:

| Axis | Metrics-only reference | Calibrated single LLM |
|---|---|---|
| Cost and speed | Best option: deterministic and free after setup | Low cost, but still API-dependent |
| Reproducibility | Fully reproducible for the same inputs | Mostly stable, but model sampling and provider behavior remain external factors |
| Issue-label substitution | Defines the reference behavior | Strong approximation after calibration: 96% recall, 90% exact match |
| Explanation | Structured but mostly rule-based | Produces a compact natural-language rationale for each verdict |
| Case-level disagreement analysis | Flags what the rules encode | Can expose when a rule may be too rigid, incomplete, or semantically questionable |
| Adaptability | Requires code-level heuristic changes | Can be redirected through prompt changes, within limits |
| Final authority | Strong for the defined deterministic benchmark | Not reliable enough to replace the full verdict policy alone |

This means the main advantage of the single LLM is not raw accuracy against the
reference system. Its advantage is interpretive and analytical. It can translate
structured heuristic evidence into a human-readable judgment, make the reasoning
surface explicit, and support case-by-case disagreement analysis. This matters
because the metrics-only reference is a designed artifact, not an infallible
oracle. When the reference and the LLM disagree, the disagreement becomes a site
for methodological inspection: the LLM may have missed a rule-supported issue,
but the deterministic rule may also be too coarse, too narrow, or insufficiently
sensitive to the semantic context of the case.

The calibrated result therefore supports a hybrid interpretation. For this
project, the metrics-only reference should remain the stable evaluation backbone,
especially for reproducible scoring and aggregate comparison. The single LLM is
best understood as a low-cost interpretive evaluator that can approximate the
reference issue taxonomy while adding readable rationales and useful signals for
manual review. It is promising as an assisted evaluation layer, but not yet as an
autonomous replacement for the complete deterministic evaluation policy.

This is enough for the final report: the project can now argue that prompt
calibration makes a small single LLM a plausible substitute for structured issue
identification, while the remaining disagreement on assessment, severity, and
recommended action demonstrates why deterministic reference systems still matter
for controlled evaluation.
