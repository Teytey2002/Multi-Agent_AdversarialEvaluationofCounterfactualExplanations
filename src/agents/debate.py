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

from agents.agents import build_debate_agents, build_single_evaluator_agent
from agents.config import LLMConfig, build_model_client, resolve_llm_config
from agents.prompts import get_issue_guidance
from agents.utils import calculate_cost, parse_judge_verdict, serialise_message


SPECIALIST_NAMES = ["Prosecutor", "Defense", "Expert_Witness"]
ALL_AGENT_NAMES  = SPECIALIST_NAMES + ["Judge"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

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
- whether the CF set as a whole provides useful guidance.

Allowed issue labels:
{issue_notes}

Debate structure:
- Up to {max_rounds} specialist rounds before the Judge gives the final verdict.
- The Judge should only speak when selected at the end.
- Base ALL arguments on the real data below — do not invent additional evidence.

Full case data:
```json
{json.dumps(case_data, indent=2)}
```
""".strip()


def _build_single_llm_prompt(case_data: dict[str, Any]) -> str:
    """Build the task prompt for the single-LLM baseline evaluator."""
    issue_notes = get_issue_guidance()

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
{json.dumps(case_data, indent=2)}
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

        judge_msgs = [t for t in transcript if t.get("source") == "Judge"]
        if not judge_msgs:
            raise RuntimeError("The Judge never produced a final message.")

        raw_judge = judge_msgs[-1]["content"]
        verdict = parse_judge_verdict(raw_judge)

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
) -> dict[str, Any]:
    """Run a single-agent baseline evaluation for one case."""

    config = llm_config or resolve_llm_config(
        provider=provider, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )
    model_client = build_model_client(config)
    evaluator = build_single_evaluator_agent(model_client)

    try:
        result = await evaluator.run(task=_build_single_llm_prompt(case_data))
        transcript = [serialise_message(m) for m in result.messages]
        if verbose:
            for item in transcript:
                print(f"[{item['source']}] {item['content']}\n")

        final = transcript[-1]["content"] if transcript else ""
        verdict = parse_judge_verdict(final)

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
        }
    finally:
        await model_client.close()


def run_single_llm(case_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper for the single-LLM baseline."""
    return asyncio.run(run_single_llm_async(case_data, **kwargs))
