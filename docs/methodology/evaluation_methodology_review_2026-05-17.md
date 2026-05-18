# Evaluation Methodology Review — 2026-05-17

**Author:** Daniel Mortada  
**Audience:** Ivan + rest of team (internal pre-meeting reference)  
**Status:** Working draft — not for submission

---

## 1. Intent

This document is my structured read of where we are empirically before our meeting today. Ivan raised a concrete architectural critique of our evaluation pipeline. I want to lay out the actual numbers, acknowledge what is correct in his critique, push back where I think the framing conflates separate problems, and propose a concrete plan for us to align on. We don't need to agree on everything today, but we do need to agree on (a) re-annotation and (b) whether to test 70B before redesigning the architecture.

Background on the deterministic baseline and heuristics is in `docs/walkthrough/session_03_*.md` and `docs/walkthrough/session_04_*.md`. This document assumes familiarity with those.

---

## 2. Where We Stand Empirically

### 2.1 Summary table

| System | Detection rate | Exact match | Avg cost/case |
|---|---|---|---|
| Metrics-only | **88.89%** (24/27 GT issues caught) | **60%** (6/10) | $0 |
| Single-LLM (8B) | 81.48% (22/27) | **60%** (6/10) | ~$0.0004 |
| Multi-agent (8B) | 70.37% (19/27) | 20% (2/10) | ~$0.0009 |

Reference: 10 cases, 4 CFs per case, 27 total ground-truth issue labels from `annotations/ground_truth_labels.json`.

The headline result is uncomfortable: the fully deterministic baseline outperforms both LLM-based approaches on detection rate, and the multi-agent approach performs worst on exact match by a wide margin (20% vs 60%).

### 2.2 Metrics-only failure pattern

- Misses `inconsistent_work_profile` in cases 0, 2, and 6.
- One false positive: `extreme_working_hours` in case 4.
- **Root cause:** The heuristic for `inconsistent_work_profile` only triggers when `workclass` is `"Without-pay"` or `"Never-worked"` with a non-null `occupation`. Our annotations apply the label more broadly — e.g., to cases where occupation and hours-per-week are internally inconsistent even with a valid workclass. This is a **taxonomy coverage gap** (the heuristic definition is narrower than the annotation intent), not a quality failure in the evaluator.

### 2.3 Multi-agent failure pattern

This is the more interesting and more serious failure. Two distinct failure modes:

**Forgetting / dropping:** The multi-agent system misses `fragile_counterfactual` in cases 0, 7, 8, and 9, despite those issues being explicitly present in the `heuristic_metrics.flagged_issues` that the Judge receives.

**Hallucinating / adding:** The system adds issues with no heuristic support:
- `fragile_counterfactual` in case 3 (not flagged by heuristics)
- `inconsistent_work_profile` + `too_many_changes` in case 8
- `inconsistent_work_profile` in case 9

The Judge's system message in `src/agents/agents.py` is explicit: *"Start from heuristic_summary.flagged_issues_union"* and *"You may add a scored issue only if it appears in at least one CF's heuristic_metrics.flagged_issues."* The model is violating both rules systematically.

**Confound — model size:** Multi-agent prompts run 13–16K tokens. Single-LLM prompts for the same model (`llama-3.1-8b-instant`) run ~3–4K tokens. Single-LLM achieves 81.48% detection. The 11-percentage-point gap between single-LLM and multi-agent, on the same model, strongly suggests the 8B model loses coherence in long context — the architecture may not be the problem; the model capacity may be.

---

## 3. What Ivan Raised

Ivan's message, verbatim:

> "Hi guys ! I went over the methodology documents that you made.
>
> First, thank you for this work 👏.
> Then I believe the problem will not be solved by changing the taxonomy.
>
> I think the problem is that the ground-truth labels, their structure, is biased towards the metrics-only baseline. Additionally, the LLM agents shouldn't output the JSON answer from scratch, producing hallucinations and forgetting labeled issue.
>
> To fix it I would do the following methodology:
>
> heuristics + metrics
>         ↓
> deterministic draft verdict
>         ↓
> LLM calibrator reviews only doubtful parts
>         ↓
> final verdict
>
> So basically making the LLM agents as an additional layer to the metrics-only baseline. Only modifying issues/labels that it flags as potentially problematic.
>
> We will discuss it tomorrow I suppose.
>
> Have a good day."

Ivan is raising two independent claims and bundling them as a single fix. I want to address them separately.

**Claim A:** The ground-truth labels are structurally biased toward metrics-only because our team wrote both.

**Claim B:** LLM agents should not generate verdicts from scratch; they should operate as a calibration layer on top of a deterministic draft, only modifying "doubtful parts."

---

## 4. My Read on Claim A — Label Bias

Ivan is correct. Our reference labels in `annotations/ground_truth_labels.json` were drafted by the same team that designed the heuristics, and the file itself says so:

```json
"annotation_status": "initial_codex_human_perspective_draft_for_team_review"
```

That phrase is not decorative — it was intentional. We knew at the time that those labels were a starting point, not ground truth. The metrics-only baseline was defined against the same conceptual framework we used when writing the annotations, so any correlation between heuristic coverage and annotation coverage is expected and real.

**However:** this is a problem with the *labels*, not the *architecture*. Ivan's proposed architectural change — making the LLM a calibrator rather than a generator — does nothing to fix label bias. A calibrator operating on top of a heuristic-derived draft would still be evaluated against labels that were influenced by the same heuristics. The bias is in the reference, not in the evaluator.

The correct fix for label bias is independent re-annotation. See Recommendation 1 below.

---

## 5. My Read on Claim B — LLM as Calibrator

The empirical data supports this concern. The multi-agent system is both forgetting flagged issues and hallucinating unflagged ones, and we can observe this because the heuristic evidence is right there in the case data the Judge receives. The "editor mode" pattern Ivan describes — LLM only proposes deltas on a pre-formed verdict — would mechanically prevent both failure modes:

- Forgetting: the deterministic draft would anchor all flagged issues; the LLM would have to explicitly argue for removal, not just omit.
- Hallucination: the LLM couldn't add an issue without the draft providing a hook for it (by design — draft only contains heuristic-supported issues).

So the proposed architecture is sound as a mechanism. My concern is what it does to the research question.

**Our current framing:** *Do LLM-based evaluators match or exceed deterministic metrics on counterfactual quality assessment?*

**Ivan's proposed framing:** *Can LLMs improve a deterministic baseline by post-processing it?*

These are different papers. The second framing is less interesting as a contribution because the answer is almost certainly "yes, a little, sometimes" for any sufficiently capable LLM. The first framing is where we have a real result — even a null result (LLMs don't beat rules here) is a legitimate finding worth reporting.

The important caveat: we don't yet know whether the multi-agent underperformance is architectural or model-size-driven. If the 8B model is just too small to handle 15K-token prompts coherently, that's not a verdict on adversarial multi-agent evaluation as a methodology. We should test a 70B model before drawing that conclusion.

---

## 6. Recommendations (in priority order)

### Recommendation 1 — Independent re-annotation (MANDATORY)

Each of the 4 team members independently annotates all 10 cases without looking at `heuristics.py`, `metrics_only.py`, or each other's labels. We then compute inter-annotator agreement — Cohen's kappa if we treat it as pairwise, Fleiss' kappa for all four raters simultaneously. We use majority vote (3/4 or 4/4 agreement) as the new reference label for each issue-case pair.

This is non-negotiable for methodology credibility. Without it, the current numbers are suspect regardless of which architecture we use. The re-annotation exercise also serves as a calibration session — it will surface genuine disagreements about taxonomy that we can then resolve explicitly.

**Effort:** ~2–3 hours per person. Should be done before any architectural change.

**Owner:** All four team members. Needs a coordination point (shared spreadsheet, or I can set up a simple annotation form).

**Deadline:** Before the next run of any evaluation system.

---

### Recommendation 2 — Re-run multi-agent on llama-3.3-70b-versatile (HIGHEST LEVERAGE TECHNICAL CHANGE)

Before redesigning the architecture, we need to know whether the multi-agent failures are caused by model size (likely, given the context-length confound) or by the architecture itself. This is a one-line change:

```powershell
$env:PYTHONPATH="src"; python scripts/run_debate.py --model llama-3.3-70b-versatile
```

The 70B model is on Groq's free tier but with tighter rate limits than the 8B. Cost per run is still in the low-single-digit cents range for 10 cases — not a budget concern. Pacing needs to be slightly more conservative than the current `--delay 70` default; I'd recommend `--delay 120` for a first pass.

If 70B achieves detection rate and exact match comparable to single-LLM (or better), the architecture is fine and the problem was always model capacity. We document that and proceed. If 70B still fails in the same ways, then Ivan's architectural critique is correct and we implement Recommendation 3.

**Effort:** ~30 minutes to run + results review.

**Owner:** Whoever has quota headroom. I can run it.

---

### Recommendation 3 — Implement Ivan's calibrator as a fourth evaluation mode (IF rec. 2 doesn't resolve the failure)

If 70B doesn't fix the multi-agent failure modes, I agree with Ivan that the calibrator pattern is the right direction. But I want to implement it as an *additive* mode, not a replacement for the adversarial debate.

Concretely: add `src/agents/calibrator.py` with a single-prompt design that takes `(case_data + deterministic draft verdict)` and outputs a structured delta:

```
{
  "keep": [...],      // issues to retain unchanged
  "add": [...],       // issues the LLM wants to add (with justification)
  "remove": [...]     // issues the LLM wants to remove (with justification)
}
```

A small adjudication layer (rules-based: accept `add` only if confidence > threshold, accept `remove` only if LLM cites specific evidence) produces the final verdict. New CLI flag: `run_debate.py --calibrator`. This extends the comparison table from 3 systems to 4 without removing any existing results.

**Effort:** ~1–2 days of coding + prompt work. Not trivial given project timeline.

**Owner:** Me, if the team agrees this is the direction. I'd want Ivan's input on the delta schema.

---

### Recommendation 4 — If time is the constraint, do only Recommendation 1

Independent re-annotation alone substantially improves the credibility of every number in our table, regardless of which architecture produces those numbers. If we're short on time before the project deadline, drop Recommendations 2 and 3 and invest all remaining time in annotation quality. A well-annotated 10-case benchmark is a better contribution than an extra system with weak reference labels.

---

## 7. Decisions We Need to Align on Today

These are the specific calls I want us to make at the meeting. They have dependencies — the order matters.

| # | Decision | Options |
|---|---|---|
| 1 | **Re-annotation process** — who does it, what tool, by when? | Shared spreadsheet / annotation form / annotate independently and submit to me |
| 2 | **Test 70B before changing architecture?** | Yes (I run it this week) / No (skip to calibrator) |
| 3 | **If 70B fails: implement calibrator as 4th mode?** | Yes, ~2 days / No, keep 3 systems and document failure as a finding |
| 4 | **Framing in the report**: how do we characterize the reference labels? | "Project artifact (human draft, team-consensus)" vs "ground truth" — we should use the former |

My position on each: yes to re-annotation (mandatory), yes to 70B test (cheap and informative), calibrator only if 70B fails, and we should explicitly call the labels a project artifact in the paper's methodology section.

---

## 8. Risks to Be Honest About

**Re-annotation by the same team may still be correlated.** Four CS engineering students who all understand the heuristics will tend to annotate in similar ways even without looking at the code. We can mitigate this by having each person annotate before any group discussion, and by using kappa to surface disagreements. We should acknowledge in the paper that our annotations are a team consensus artifact, not an external gold standard.

**70B may not be enough.** The 70B model handles long context better, but if the failure is architectural (e.g., the adversarial roles actually create noise rather than reducing it, or the AutoGen turn-selection is causing information loss between turns), increasing model size won't fix it. In that case, Ivan's calibrator proposal is not just a patch but a structural rethink of what the multi-agent system is supposed to do.

**Adding a 4th mode increases scope in a time-constrained project.** We have an existing codebase that already supports three evaluation modes. Adding a fourth that is architecturally different (calibrator vs. generator) means new prompts, a new output schema, new scoring logic, and new visualization. That's probably 2–3 days of careful work. If the project deadline is close, this might not be feasible. We should set a go/no-go point: if the 70B test is done by end of this week and the result is still poor, we have ~1 week to implement the calibrator; otherwise we document the failure and move on.

**The "editor mode" changes the research question in a way that may not be reportable as a positive result.** If the calibrator works, the story becomes "rules + LLM > rules alone" — which is a narrow contribution. If it doesn't work, we've spent a week finding out. We should agree upfront on what "success" looks like for the calibrator before committing to build it.

---

*End of document. Bring questions to the meeting.*
