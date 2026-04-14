"""
Utility functions for the debate system.

- Message serialisation and content extraction
- Judge verdict parsing (JSON from freeform text)
- Token / cost estimation
- Transcript saving (Markdown)
- Agreement computation (verdicts vs ground truth)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.config import MODEL_PRICING_USD_PER_1M, DEFAULT_MODELS


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_content_text(message: Any) -> str:
    """Best-effort conversion of an AutoGen message (or dict) into plain text."""
    if isinstance(message, str):
        return message

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                chunks.append(json.dumps(item, ensure_ascii=True))
            else:
                chunks.append(str(item))
        return "\n".join(c for c in chunks if c)

    return str(content)


def serialise_message(message: Any) -> dict[str, Any]:
    """Convert an AutoGen message into a JSON-friendly transcript record."""
    usage = getattr(message, "models_usage", None)
    prompt_tokens      = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens  = getattr(usage, "completion_tokens", None) if usage else None

    return {
        "source":            getattr(message, "source", "unknown"),
        "type":              message.__class__.__name__,
        "content":           extract_content_text(message),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _extract_first_json_block(text: str) -> str | None:
    """Extract the first balanced JSON object from free-form text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_judge_verdict(message: Any) -> dict[str, Any]:
    """
    Extract the Judge's JSON verdict from a message.

    Supports raw JSON, fenced ``json`` blocks, and inline JSON objects.
    """
    raw_text = extract_content_text(message).replace("VERDICT_COMPLETE", "").strip()
    if not raw_text:
        raise ValueError("Judge message was empty; no verdict to parse.")

    candidates: list[str] = [raw_text]

    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(fenced)

    generic = re.findall(r"```\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    candidates.extend(generic)

    inline = _extract_first_json_block(raw_text)
    if inline:
        candidates.append(inline)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            verdict = json.loads(candidate)
            verdict.setdefault("flagged_issues", [])
            if not isinstance(verdict["flagged_issues"], list):
                verdict["flagged_issues"] = [str(verdict["flagged_issues"])]
            verdict["flagged_issues"] = [str(i) for i in verdict["flagged_issues"]]
            return verdict
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ValueError(f"Could not parse Judge verdict as JSON. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def calculate_cost(
    messages: list[dict[str, Any]] | list[Any],
    model_name: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Estimate token usage and cost from transcript messages."""
    prompt_tokens = 0
    completion_tokens = 0
    estimated_completion_tokens = 0

    for message in messages:
        if isinstance(message, dict):
            pv = message.get("prompt_tokens")
            cv = message.get("completion_tokens")
            content_text = extract_content_text(message)
        else:
            usage = getattr(message, "models_usage", None)
            pv = getattr(usage, "prompt_tokens", None) if usage else None
            cv = getattr(usage, "completion_tokens", None) if usage else None
            content_text = extract_content_text(message)

        if pv is not None:
            prompt_tokens += int(pv)
        if cv is not None:
            completion_tokens += int(cv)
        elif content_text:
            estimated_completion_tokens += max(1, len(content_text) // 4)

    total_completion = completion_tokens + estimated_completion_tokens
    total_tokens = prompt_tokens + total_completion

    if not model_name and provider and provider in DEFAULT_MODELS:
        model_name = DEFAULT_MODELS[provider]

    pricing = MODEL_PRICING_USD_PER_1M.get(model_name or "", {"input": 0.0, "output": 0.0})
    cost = (
        prompt_tokens     / 1_000_000 * pricing["input"]
        + total_completion / 1_000_000 * pricing["output"]
    )

    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6),
        "used_fallback_estimate": estimated_completion_tokens > 0 and completion_tokens == 0,
    }


# ---------------------------------------------------------------------------
# Transcript I/O
# ---------------------------------------------------------------------------

def save_debate_transcript(
    case_id: int,
    transcript: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Save a human-readable Markdown transcript to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"case_{case_id:02d}_transcript.md"

    lines = [f"# Case {case_id} — Debate Transcript", ""]
    for item in transcript:
        speaker = item.get("source", "unknown")
        content = str(item.get("content", "")).strip()
        pt = item.get("prompt_tokens")
        ct = item.get("completion_tokens")

        lines.append(f"## {speaker}")
        if pt is not None or ct is not None:
            lines.append(f"_usage: prompt_tokens={pt}, completion_tokens={ct}_")
        lines.append("")
        lines.append(content if content else "<empty>")
        lines.append("")

    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def compute_agreement(
    verdicts: list[dict[str, Any]],
    ground_truth: list[list[str]] | list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare verdicts against planted / expected defects.

    Returns percentage-style metrics on a 0-100 scale.
    """
    if not verdicts:
        return {
            "total_cases": 0, "cases_with_issues": 0, "clean_cases": 0,
            "total_ground_truth_issues": 0, "caught_issues": 0,
            "false_positive_clean_cases": 0,
            "detection_rate": 0.0, "false_positive_rate": 0.0, "exact_match_rate": 0.0,
            "per_case": [],
        }

    if ground_truth and isinstance(ground_truth[0], dict):
        gt_lists = [case.get("ground_truth_issues", []) for case in ground_truth]  # type: ignore[union-attr]
    else:
        gt_lists = ground_truth  # type: ignore[assignment]

    per_case: list[dict[str, Any]] = []
    total_gt = caught = clean = fp_clean = exact = with_issues = 0

    for verdict, truth_list in zip(verdicts, gt_lists):
        flagged = set(str(i) for i in verdict.get("flagged_issues", []))
        truth   = set(str(i) for i in truth_list)

        total_gt += len(truth)
        caught   += len(flagged & truth)

        if truth:
            with_issues += 1
        else:
            clean += 1
            if flagged:
                fp_clean += 1

        missed = sorted(truth - flagged)
        extra  = sorted(flagged - truth)
        match  = not missed and not extra
        if match:
            exact += 1

        per_case.append({
            "case_id": verdict.get("case_id"),
            "match": match,
            "missed_issues": missed,
            "extra_issues": extra,
        })

    n = len(verdicts)
    return {
        "total_cases":               n,
        "cases_with_issues":         with_issues,
        "clean_cases":               clean,
        "total_ground_truth_issues": total_gt,
        "caught_issues":             caught,
        "false_positive_clean_cases": fp_clean,
        "detection_rate":      round(caught / total_gt * 100 if total_gt else 100.0, 2),
        "false_positive_rate": round(fp_clean / clean  * 100 if clean    else 0.0,   2),
        "exact_match_rate":    round(exact   / n       * 100,                         2),
        "per_case": per_case,
    }
