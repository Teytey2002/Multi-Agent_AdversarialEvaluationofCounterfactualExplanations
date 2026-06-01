"""
Debate orchestration — runs multi-agent or single-LLM evaluation on a case.

Adapted from the AutoGen PoC.  Key changes vs PoC:
- ``_build_case_prompt`` now handles the enriched multi-CF case schema
  (``counterfactuals[]`` array, ``metrics``, ``model_info``, etc.).
- Issue taxonomy is loaded from ``agents.prompts`` (placeholder until Ivan delivers).
- No dependency on mock_data — cases come from ``results/cases.json`` via case_builder.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat

from agents.agents import (
    SPECIALIST_OUTPUT_PROTOCOL,
    build_debate_agents,
    build_single_evaluator_agent,
    build_single_explainer_agent,
)
from agents.config import LLMConfig, build_model_client, resolve_llm_config
from agents.prompts import get_issue_guidance
from agents.utils import calculate_cost, parse_judge_verdict, serialise_message


SPECIALIST_NAMES = ["Prosecutor", "Defense", "Expert_Witness"]
ALL_AGENT_NAMES  = SPECIALIST_NAMES + ["Judge"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep numeric evidence compact enough for low-token smoke tests."""
    compact: dict[str, Any] = {}
    for label, items in evidence.items():
        compact_items = []
        for item in items[:3]:
            if isinstance(item, dict):
                compact_items.append({
                    key: value
                    for key, value in item.items()
                    if key != "reason"
                })
            else:
                compact_items.append(item)
        compact[label] = compact_items
    return compact


def _compact_heuristic_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the heuristic fields agents need without verbose repeated prose."""
    return {
        "changed_features": metrics.get("changed_features", []),
        "actionable_sparsity": metrics.get("actionable_sparsity"),
        "burden_count": metrics.get("burden_count"),
        "flagged_issues": metrics.get("flagged_issues", []),
        "constraint_violations": metrics.get("constraint_violations", []),
        "issue_evidence": _compact_evidence(metrics.get("issue_evidence", {})),
    }


def _compact_case_for_prompt(case_data: dict[str, Any]) -> dict[str, Any]:
    """
    Drop redundant full CF feature rows before sending cases to an LLM.

    The original case JSON stores full feature dictionaries for every CF plus
    repeated explanatory evidence. That is useful as an artifact, but it exceeds
    small free-tier model limits. The prompt only needs the original row, per-CF
    changes, confidence, metrics, and heuristic flags/evidence. Ground-truth
    labels are deliberately excluded because they are used only for scoring
    after the LLM has produced a verdict.
    """
    counterfactuals = []
    for cf in case_data.get("counterfactuals", []):
        counterfactuals.append({
            "cf_rank": cf.get("cf_rank"),
            "cf_confidence": cf.get("cf_confidence"),
            "features_changed": cf.get("features_changed", []),
            "changes_summary": cf.get("changes_summary", {}),
            "heuristic_metrics": _compact_heuristic_metrics(
                cf.get("heuristic_metrics", {})
            ),
        })

    heuristic_summary = case_data.get("heuristic_summary", {})

    return {
        "case_id": case_data.get("case_id"),
        "domain": case_data.get("domain"),
        "model_info": case_data.get("model_info", {}),
        "original": case_data.get("original", {}),
        "prediction": case_data.get("prediction"),
        "prediction_confidence": case_data.get("prediction_confidence"),
        "true_label": case_data.get("true_label"),
        "is_false_negative": case_data.get("is_false_negative"),
        "generation_policy": case_data.get("generation_policy", {}),
        "metrics": case_data.get("metrics", {}),
        "heuristic_summary": {
            "flagged_issues_union": heuristic_summary.get("flagged_issues_union", []),
            "constraint_violations_union": heuristic_summary.get(
                "constraint_violations_union", []
            ),
        },
        "counterfactuals": counterfactuals,
    }


def _build_case_prompt(case_data: dict[str, Any], max_rounds: int) -> str:
    """
    Build the opening task prompt for a multi-agent debate.

    The case_data dict follows the schema produced by ``case_builder.py``:
    - ``counterfactuals`` is an **array** (multiple CFs per individual).
    - ``metrics`` contains real DiCE quality metrics.
    - ``model_info`` includes the classifier's evaluation scores.
    - ``changes_summary`` per CF shows {feature: {from, to}}.
    """
    issue_notes = get_issue_guidance()
    prompt_case = _compact_case_for_prompt(case_data)

    # Build a concise human-readable summary on top of the raw JSON.
    original = case_data.get("original", {})
    n_cfs    = len(case_data.get("counterfactuals", []))
    pred     = case_data.get("prediction", "?")
    conf     = case_data.get("prediction_confidence", "?")
    true_lbl = case_data.get("true_label", "?")
    fn_flag  = case_data.get("is_false_negative", False)

    header_lines = [
        f"Individual: {original.get('age', '?')}yo {original.get('sex', '?')}, "
        f"{original.get('occupation', '?')}, {original.get('education', '?')}, "
        f"{original.get('marital-status', '?')}",
        f"Model prediction: {pred} (confidence {conf})",
        f"True label: {true_lbl}  |  False negative: {fn_flag}",
        f"Number of counterfactuals generated: {n_cfs}",
    ]
    header = "\n".join(header_lines)

    return f"""
You are participating in a fairness review of counterfactual explanations generated
by a Logistic Regression model on the Adult Income dataset.

{header}

Focus your analysis on:
- fairness (were immutable/protected features changed?),
- feasibility (are the suggested changes realistic for this individual?),
- actionability (can the person actually make these changes?),
- quality metrics (sparsity, proximity, diversity, confidence),
- deterministic heuristic evidence (counterfactuals[*].heuristic_metrics and heuristic_summary),
- whether the CF set as a whole provides useful guidance.

Allowed issue labels:
{issue_notes}

Debate structure:
- Up to {max_rounds} specialist rounds before the Judge gives the final verdict.
- The Judge should only speak when selected at the end.
- Base ALL arguments on the real data below - do not invent additional evidence.

Specialist response format for Prosecutor, Defense, and Expert_Witness only:
{SPECIALIST_OUTPUT_PROTOCOL}

Judge response format:
- Ignore the specialist format.
- Return exactly one fenced JSON verdict, followed by the completion sentinel
  specified in the Judge system message.

Case data:
```json
{json.dumps(prompt_case, separators=(",", ":"))}
```
""".strip()


def _build_single_llm_prompt(case_data: dict[str, Any]) -> str:
    """Build the task prompt for the single-LLM baseline evaluator."""
    issue_notes = get_issue_guidance()
    prompt_case = _compact_case_for_prompt(case_data)

    return f"""
Review the following counterfactual explanations generated for an income-prediction
model and decide whether the set is fair, unfair, or ambiguous.

The case contains multiple counterfactuals per individual, each with confidence
scores and a changes_summary showing what was modified.  Real DiCE quality metrics
(validity, proximity, sparsity, diversity) are included.

Allowed issue labels:
{issue_notes}

Return the exact JSON schema requested in your system message.

Case data:
```json
{json.dumps(prompt_case, separators=(",", ":"))}
```
""".strip()


def _compact_case_for_explanation(case_data: dict[str, Any]) -> dict[str, Any]:
    """Return only the evidence needed to explain an already-fixed verdict."""
    compact = _compact_case_for_prompt(case_data)
    return {
        "case_id": compact.get("case_id"),
        "original": compact.get("original", {}),
        "prediction": compact.get("prediction"),
        "prediction_confidence": compact.get("prediction_confidence"),
        "true_label": compact.get("true_label"),
        "is_false_negative": compact.get("is_false_negative"),
        "metrics": compact.get("metrics", {}),
        "heuristic_summary": compact.get("heuristic_summary", {}),
        "counterfactuals": compact.get("counterfactuals", []),
    }


def _build_single_explanation_prompt(
    case_data: dict[str, Any],
    verdict: dict[str, Any],
) -> str:
    """Build the second-stage prompt that explains a fixed single-LLM verdict."""
    compact_case = _compact_case_for_explanation(case_data)
    heuristic_issues = set(
        compact_case.get("heuristic_summary", {}).get("flagged_issues_union", [])
    )
    verdict_issues = set(str(i) for i in verdict.get("flagged_issues", []))
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

    return f"""
Explain the fixed single-LLM evaluation below for a non-expert reader.

Important:
- The evaluation is about the counterfactual explanation set, not whether the original model prediction is morally unfair.
- Do not revise the fixed evaluation.
- Do not add or remove issue labels.
- Mention disagreement risk only if `issue_alignment` lists missed or extra issues.
- Put the complete explanation inside `expert_explanation`.
- Return only the JSON schema requested by your system message.

Input:
```json
{json.dumps(explanation_input, separators=(",", ":"))}
```
""".strip()


# ---------------------------------------------------------------------------
# Speaker selection helpers
# ---------------------------------------------------------------------------

def _speaker_history(messages: Sequence[Any]) -> list[str]:
    return [
        getattr(m, "source", "")
        for m in messages
        if getattr(m, "source", "") in ALL_AGENT_NAMES
    ]


def _build_round_robin_selector(max_rounds: int):
    def selector_func(messages: Sequence[Any]) -> str | None:
        history = _speaker_history(messages)
        specialist_turns = sum(1 for s in history if s in SPECIALIST_NAMES)
        if specialist_turns >= max_rounds * len(SPECIALIST_NAMES):
            return "Judge"
        return SPECIALIST_NAMES[specialist_turns % len(SPECIALIST_NAMES)]
    return selector_func


def _build_auto_candidate_func(max_rounds: int):
    def candidate_func(messages: Sequence[Any]) -> list[str]:
        history = _speaker_history(messages)
        specialist_turns = sum(1 for s in history if s in SPECIALIST_NAMES)
        if specialist_turns == 0:
            return ["Prosecutor"]
        if specialist_turns >= max_rounds * len(SPECIALIST_NAMES):
            return ["Judge"]
        candidates = list(SPECIALIST_NAMES)
        prev = history[-1] if history else None
        if prev in candidates and len(candidates) > 1:
            candidates.remove(prev)
        return candidates
    return candidate_func


# ---------------------------------------------------------------------------
# Multi-agent debate
# ---------------------------------------------------------------------------

async def run_debate_async(
    case_data: dict[str, Any],
    *,
    llm_config: LLMConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
    speaker_selection: str = "round_robin",
    max_rounds: int = 2,
    temperature: float = 0.2,
    max_tokens: int = 700,
    turn_delay: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a multi-agent debate on a single case and return the result dict."""

    config = llm_config or resolve_llm_config(
        provider=provider, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )
    model_client = build_model_client(config)
    agents = build_debate_agents(model_client)

    selector_prompt = """
Select the single best next speaker for this counterfactual fairness debate.

{roles}

Conversation so far:
{history}

Choose exactly one participant from {participants}.
Return only the participant name.
""".strip()

    termination = (
        TextMentionTermination("VERDICT_COMPLETE")
        | MaxMessageTermination(max_messages=max_rounds * len(SPECIALIST_NAMES) + 4)
    )

    if speaker_selection == "round_robin":
        team = SelectorGroupChat(
            participants=list(agents.values()),
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            selector_func=_build_round_robin_selector(max_rounds),
            allow_repeated_speaker=True,
        )
    elif speaker_selection == "auto":
        team = SelectorGroupChat(
            participants=list(agents.values()),
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            candidate_func=_build_auto_candidate_func(max_rounds),
        )
    else:
        await model_client.close()
        raise ValueError("speaker_selection must be 'round_robin' or 'auto'.")

    transcript: list[dict[str, Any]] = []
    stop_reason = "unknown"

    try:
        stream = team.run_stream(task=_build_case_prompt(case_data, max_rounds))
        async for event in stream:
            if hasattr(event, "messages") and hasattr(event, "stop_reason"):
                stop_reason = getattr(event, "stop_reason", "unknown")
                continue
            entry = serialise_message(event)
            transcript.append(entry)
            if verbose:
                print(f"[{entry['source']}] {entry['content']}\n")
            if (
                turn_delay > 0
                and entry.get("source") in SPECIALIST_NAMES
            ):
                await asyncio.sleep(turn_delay)

        judge_msgs = [t for t in transcript if t.get("source") == "Judge"]
        if not judge_msgs:
            raise RuntimeError("The Judge never produced a final message.")

        raw_judge = ""
        verdict = None
        last_error: Exception | None = None
        for judge_msg in reversed(judge_msgs):
            candidate = str(judge_msg.get("content", "")).strip()
            if not candidate:
                continue
            try:
                verdict = parse_judge_verdict(candidate)
                raw_judge = candidate
                break
            except ValueError as exc:
                last_error = exc

        if verdict is None:
            snippet = str(judge_msgs[-1].get("content", ""))[:300]
            raise ValueError(
                "No parseable Judge verdict found. "
                f"Last parse error: {last_error}. Last Judge snippet: {snippet!r}"
            )

        return {
            "case_id":            case_data["case_id"],
            "speaker_selection":  speaker_selection,
            "transcript":         transcript,
            "verdict":            verdict,
            "raw_verdict_message": raw_judge,
            "stop_reason":        stop_reason,
            "cost":               calculate_cost(transcript, model_name=config.model, provider=config.provider),
            "model":              config.model,
            "provider":           config.provider,
        }
    finally:
        await model_client.close()


def run_debate(case_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper — ``run_debate(case)`` for simple scripts."""
    return asyncio.run(run_debate_async(case_data, **kwargs))


# ---------------------------------------------------------------------------
# Single-LLM baseline
# ---------------------------------------------------------------------------

async def run_single_llm_async(
    case_data: dict[str, Any],
    *,
    llm_config: LLMConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
    verbose: bool = False,
    include_explainability: bool = False,
) -> dict[str, Any]:
    """Run a single-agent baseline evaluation for one case."""

    config = llm_config or resolve_llm_config(
        provider=provider, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )
    model_client = build_model_client(config)
    evaluator = build_single_evaluator_agent(model_client)
    explainer = (
        build_single_explainer_agent(model_client)
        if include_explainability
        else None
    )

    try:
        result = await evaluator.run(task=_build_single_llm_prompt(case_data))
        transcript = [serialise_message(m) for m in result.messages]
        if verbose:
            for item in transcript:
                print(f"[{item['source']}] {item['content']}\n")

        final = transcript[-1]["content"] if transcript else ""
        verdict = parse_judge_verdict(final)

        if explainer is not None:
            explanation_result = await explainer.run(
                task=_build_single_explanation_prompt(case_data, verdict)
            )
            explanation_transcript = [
                serialise_message(m) for m in explanation_result.messages
            ]
            transcript.extend(explanation_transcript)
            explanation_final = (
                explanation_transcript[-1]["content"]
                if explanation_transcript
                else ""
            )
            explanation_payload = parse_judge_verdict(explanation_final)
            verdict["expert_explanation"] = str(
                explanation_payload.get("expert_explanation", "")
            ).strip()

        return {
            "case_id":            case_data["case_id"],
            "speaker_selection":  "single_llm",
            "transcript":         transcript,
            "verdict":            verdict,
            "raw_verdict_message": final,
            "stop_reason":        "completed",
            "cost":               calculate_cost(transcript, model_name=config.model, provider=config.provider),
            "model":              config.model,
            "provider":           config.provider,
            "explainability":     include_explainability,
        }
    finally:
        await model_client.close()


def run_single_llm(case_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper for the single-LLM baseline."""
    return asyncio.run(run_single_llm_async(case_data, **kwargs))
