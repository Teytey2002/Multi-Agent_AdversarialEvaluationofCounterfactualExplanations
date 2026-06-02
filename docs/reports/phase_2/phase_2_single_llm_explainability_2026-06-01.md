# Phase 2 Single-LLM Explainability Layer Results

Date: 2026-06-01
Model: `llama-3.1-8b-instant` via Groq
Mode: single-LLM only, no multi-agent rerun
Reference system: `results/metrics_only_outputs/metrics_only_latest.json`
Final run: `results/debate_outputs/llama-3.1-8b-instant/single_llm_20260601_113616/single_llm_results.json`
Final substitution score: `results/substitution_outputs/substitution_scores_20260601_114801.json`

---

## 1. Purpose

This report documents the Phase 2 explainability extension added after the
single-LLM prompt calibration pass.

The motivation was simple: the calibrated single LLM had become strong at
approximating the metrics-only reference issue labels, but the JSON field
`reasoning_summary` remained too compressed. It usually restated the taxonomy
labels rather than explaining the case in a way a non-expert reader could
understand.

The goal was therefore not to chase higher substitution scores. The goal was to
add an interpretive layer that explains:

1. what the original prediction and counterfactuals were;
2. which deterministic heuristic evidence mattered;
3. why the fixed issue labels were selected;
4. why the final action follows;
5. whether there is any disagreement risk between the fixed verdict and the
   deterministic heuristic evidence.

---

## 2. Design Decision

A first design was tested where the single LLM produced both the verdict and a
longer explanation in the same response. A one-case smoke test on case 7 showed
that this was unsafe: the model produced an `expert_explanation`, but the extra
explanation burden caused it to omit `fragile_counterfactual` from
`flagged_issues`, even though that label was present in the heuristic evidence.

We therefore rejected the single-pass design.

The final implementation is a two-stage, toggleable layer:

1. the calibrated `Single_Evaluator` first produces the normal scored verdict;
2. a separate `Explanation_Layer` then explains that fixed verdict;
3. the explanation layer is explicitly forbidden from changing issue labels,
   severity, assessment, confidence, or recommended action;
4. the parsed `expert_explanation` is attached to the already-produced verdict.

This is the important methodological point: explainability is added after the
scored decision, so it does not become another hidden prompt constraint on issue
selection.

---

## 3. Code Evidence

The feature is exposed through a CLI toggle:

```powershell
$env:PYTHONPATH="src"; python scripts/run_debate.py --single-llm --explainability --max-tokens 700
```

The runner stores the toggle in the output config:

```json
{
  "speaker_selection": "single_llm",
  "turn_delay": 0,
  "explainability": true
}
```

The second-stage prompt is built around a fixed evaluation object rather than an
open-ended decision task:

```python
explanation_input = {
    "evaluation_scope": "counterfactual_explanation_set_not_original_prediction",
    "case_evidence": compact_case,
    "fixed_cf_evaluation": {
        "case_id": verdict.get("case_id"),
        "cf_set_assessment": verdict.get("overall_assessment"),
        "flagged_issues": sorted(verdict_issues),
        "severity": verdict.get("severity"),
        "confidence": verdict.get("confidence"),
        "reasoning_summary": verdict.get("reasoning_summary"),
        "recommended_action": verdict.get("recommended_action"),
    },
    "issue_alignment": {
        "missed_heuristic_issues": sorted(heuristic_issues - verdict_issues),
        "extra_verdict_issues": sorted(verdict_issues - heuristic_issues),
        "exact_issue_alignment": heuristic_issues == verdict_issues,
    },
}
```

The explanation system prompt also fixes the scope:

```text
Explain an already computed single-LLM verdict about the counterfactual
explanation set; do not revise the verdict.
Never say the original model prediction itself is unfair; the verdict concerns
the counterfactual explanation set.
Mention disagreement risk only when the supplied issue-alignment facts list
missed or extra issues.
```

---

## 4. Validation

The regression suite passed after the implementation:

```text
Ran 25 tests in 0.031s
OK
```

The new tests verify that:

- the default single-LLM verdict prompt remains unchanged and does not request
  `expert_explanation`;
- the explanation prompt excludes legacy annotation fields from the case data;
- the explanation prompt receives the fixed verdict and issue-alignment facts;
- `parse_judge_verdict()` preserves `expert_explanation` when present.

The final full run completed all 10 cases:

```text
Mode:     single_llm
Model:    llama-3.1-8b-instant
Explain:  enabled
Successful: 10/10
```

Output QA:

```json
{
  "total": 10,
  "missing_explanations": 0,
  "overclaim_matches": 0,
  "total_cost": 0.004954,
  "avg_explanation_chars": 691.0
}
```

Here `overclaim_matches` is a simple string check for obvious phrases such as
"original prediction is unfair" or "unfair prediction". It is not a full
semantic quality metric, but it catches the main wording error found during
smoke testing.

---

## 5. Substitution Results

The final explainability run was scored against the metrics-only reference:

Metrics use the unified Phase 1 vocabulary (precision / recall / F1), each
reference issue label treated as the positive class. `recall` is the measure
previously called `detection_rate`.

| System | Precision% | Recall% | F1% | Exact match | Assess agree | Severity agree | Action agree | Perfect cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibrated single LLM, no explanation | 100.0 | 96.0 | 98.0 | 90.0 | 50.0 | 50.0 | 60.0 | 9 / 10 |
| Explainability layer run | 96.0 | 96.0 | 96.0 | 80.0 | 50.0 | 50.0 | 60.0 | 8 / 10 |

Recall stayed at 96%, while precision dipped from 100% to 96% because the fresh
run added one extra label (so F1 moved from 98% to 96%). Exact match decreased
from 90% to 80%, because this was a fresh LLM run and the first-stage verdict
varied on two cases:

| Case | Difference vs metrics-only reference |
|---:|---|
| 0 | Extra `extreme_working_hours` |
| 2 | Missed `fragile_counterfactual` |

This should not be interpreted as the explanation layer directly changing the
labels. In the final two-stage design, the explanation is generated after the
verdict is fixed. The difference is better understood as normal stochastic
variation in a fresh single-LLM verdict run.

---

## 6. Cost Impact

The explainability layer adds a second LLM call per case, so cost increases.

| Run | Total cost | Average cost/case |
|---|---:|---:|
| Calibrated single LLM, no explanation | $0.002955 | $0.000296 |
| Explainability layer run | $0.004954 | $0.000495 |

The absolute cost remains very small, but the relative increase is meaningful:
about 67% more cost per case. This is the main practical tradeoff of the
explainability layer.

---

## 7. Example Output

Case 7 is a useful example because it previously exposed fragility sensitivity.
The final explainability run preserved the full issue set:

```json
{
  "case_id": 7,
  "overall_assessment": "unfair",
  "flagged_issues": [
    "fragile_counterfactual",
    "implausible_time_dependent_change",
    "too_many_changes"
  ],
  "severity": "high",
  "recommended_action": "review",
  "expert_explanation": "The original prediction was that the individual would earn <=50K. The counterfactual explanation set proposed changes to the individual's workclass, occupation, capital loss, and hours per week. However, these changes were deemed unrealistic, particularly the modification of education_num, which barely reached the favorable class. Furthermore, the counterfactuals proposed too many changes at once, placing an unrealistic burden on the individual. As a result, the counterfactual explanation set was assessed as unfair. The issue set aligns with deterministic heuristic evidence, indicating no disagreement risk. The recommended action is to review the counterfactual explanation set."
}
```

This explanation is more informative than the short `reasoning_summary`, because
it gives a reader the structure of the case and explains why the fixed verdict
was reached. It is still not perfect: the wording around education and
confidence remains compressed, and the explanation is generated by an 8B model.
Therefore it should be treated as an interpretive aid, not as a new authority.

---

## 8. Interpretation

The main finding is that explainability adds qualitative value, not scoring
value.

On issue-label substitution, the calibrated single LLM was already strong. The
explainability layer did not improve aggregate recall, and the final
fresh run had slightly lower exact-match agreement because of normal LLM
variation. The layer's value is instead that it makes each verdict easier to
inspect, especially when the single LLM and the metrics-only reference disagree.

This fits the Phase 2 methodology. The metrics-only reference remains the stable
benchmark for aggregate scoring. The single LLM remains useful as an
interpretive assistant around that benchmark. The explanation layer strengthens
that assistant role by turning a compact verdict into a readable case analysis.

The most defensible conclusion is therefore:

> The toggleable explainability layer should be kept as an optional qualitative
> analysis mode. It should not replace substitution scoring, and it should not be
> used to claim better label agreement. Its contribution is better auditability,
> clearer case-level review, and more accessible communication of why a verdict
> was produced.

---

## 9. Limitations

Three limitations remain.

First, the explanation layer costs more because it uses a second LLM call per
case.

Second, explanations are still model-generated. Even with stronger prompt
constraints, they require qualitative inspection. The string-level QA check is a
basic safeguard, not a complete factuality guarantee.

Third, the scalar fields remain only partially aligned with the metrics-only
reference. Assessment agreement stayed at 50%, while severity and recommended
action agreement stayed at 50% and 60%. This confirms the previous finding:
issue-label substitution is easier than full verdict-policy substitution.

---

## 10. Practical Recommendation

For the final project, use three modes distinctly:

1. metrics-only reference for reproducible scoring;
2. calibrated single LLM for low-cost issue-label substitution;
3. single LLM with `--explainability` for case-level qualitative analysis.

Do not use the explainability layer as the main quantitative result. Use it to
support the report's discussion of interpretability, disagreement analysis, and
human-facing auditability.
