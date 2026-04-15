"""
run_debate.py — CLI entry point for running multi-agent or single-LLM
counterfactual evaluations on real pipeline cases.

Usage examples
--------------
# Multi-agent debate (default: groq / llama-3.1-8b-instant, round_robin)
$env:PYTHONPATH="src"; python src/run_debate.py

# Single-LLM baseline
$env:PYTHONPATH="src"; python src/run_debate.py --single-llm

# Specific provider / model
$env:PYTHONPATH="src"; python src/run_debate.py --provider gemini --model gemini-2.5-flash

# Run only cases 0 and 2
$env:PYTHONPATH="src"; python src/run_debate.py --case-ids 0 2

# Verbose + auto speaker selection
$env:PYTHONPATH="src"; python src/run_debate.py --speaker-selection auto --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.config import resolve_llm_config
from agents.debate import run_debate, run_single_llm
from agents.utils import compute_agreement, save_debate_transcript


PROJECT_ROOT = Path(__file__).resolve().parents[1]      # repo root
CASES_PATH   = PROJECT_ROOT / "results" / "cases.json"
OUTPUT_ROOT  = PROJECT_ROOT / "results" / "debate_outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_slug(model_name: str) -> str:
    """Filesystem-safe slug from a model name."""
    slug = model_name.rsplit("/", 1)[-1]
    slug = re.sub(r'[<>:"/\\|?*]', "-", slug)
    return slug


def _format_issues(issues: list[str]) -> str:
    return ", ".join(issues) if issues else "clean"


def _format_verdict(verdict: dict[str, Any] | None) -> str:
    if not verdict:
        return "ERROR"
    assessment = verdict.get("overall_assessment", "?")
    issues = verdict.get("flagged_issues", [])
    return f"{assessment} ({_format_issues(list(issues))})"


def _print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("  (no results)")
        return

    headers = ["Case", "GT Issues", "Verdict", "Match", "Cost"]
    widths  = [len(h) for h in headers]
    keys    = ["case_id", "ground_truth", "verdict", "match", "cost"]
    for row in rows:
        for i, k in enumerate(keys):
            widths[i] = max(widths[i], len(row[k]))

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*(row[k] for k in keys)))


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_cases(path: Path, case_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Load cases from JSON, optionally filtering by case_id."""
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if case_ids is not None:
        id_set = set(case_ids)
        cases = [c for c in cases if c["case_id"] in id_set]
        if not cases:
            raise ValueError(f"No cases matched --case-ids {case_ids}")
    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multi-agent debate or single-LLM evaluation on real CF cases.",
    )
    p.add_argument("--provider", choices=["groq", "gemini", "openai"], default=None,
                   help="LLM provider (default: env var or groq).")
    p.add_argument("--model", default=None,
                   help="Model override (e.g. llama-3.3-70b-versatile).")
    p.add_argument("--speaker-selection", choices=["round_robin", "auto"], default="round_robin",
                   help="Speaker strategy for the multi-agent debate.")
    p.add_argument("--max-rounds", type=int, default=2,
                   help="Specialist rounds before the Judge speaks.")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--single-llm", action="store_true",
                   help="Run single-LLM baseline instead of debate.")
    p.add_argument("--case-ids", nargs="+", type=int, default=None,
                   help="Run only specific case IDs (e.g. --case-ids 0 2 5).")
    p.add_argument("--cases-file", type=str, default=str(CASES_PATH),
                   help=f"Path to cases JSON (default: {CASES_PATH}).")
    p.add_argument("--delay", type=int, default=None,
                   help="Seconds between cases for rate-limit cooldown.")
    p.add_argument("--verbose", action="store_true",
                   help="Print agent messages to console.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    mode = "single_llm" if args.single_llm else "multi_agent"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Resolve LLM config
    llm_config = resolve_llm_config(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    slug      = _model_slug(llm_config.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = OUTPUT_ROOT / slug / f"{mode}_{timestamp}"
    tx_dir    = run_dir / "transcripts"
    tx_dir.mkdir(parents=True, exist_ok=True)

    # Load real cases
    cases = load_cases(Path(args.cases_file), case_ids=args.case_ids)

    # Resolve inter-case delay
    default_delays = {"gemini": 60, "groq": 10, "openai": 0}
    delay = args.delay if args.delay is not None else default_delays.get(llm_config.provider, 0)

    print(f"Mode:     {mode}")
    print(f"Provider: {llm_config.provider}")
    print(f"Model:    {llm_config.model}")
    if not args.single_llm:
        print(f"Speaker:  {args.speaker_selection}")
    print(f"Cases:    {len(cases)} (IDs: {[c['case_id'] for c in cases]})")
    if delay:
        print(f"Delay:    {delay}s between cases")
    print()

    case_results: list[dict[str, Any]] = []
    table_rows:   list[dict[str, str]] = []
    verdicts:     list[dict[str, Any]] = []
    gt_lists:     list[list[str]]      = []
    failures = 0

    for idx, case in enumerate(cases):
        if idx > 0 and delay > 0:
            print(f"  … waiting {delay}s for rate-limit cooldown …")
            time.sleep(delay)

        print(f"{'─'*40} Case {case['case_id']} {'─'*40}")
        try:
            if args.single_llm:
                result = run_single_llm(
                    case, llm_config=llm_config,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    verbose=args.verbose,
                )
            else:
                result = run_debate(
                    case, llm_config=llm_config,
                    speaker_selection=args.speaker_selection,
                    max_rounds=args.max_rounds,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    verbose=args.verbose,
                )

            tx_path = save_debate_transcript(
                case_id=case["case_id"],
                transcript=result["transcript"],
                output_dir=tx_dir,
            )

            verdict  = result["verdict"]
            gt       = case.get("ground_truth_issues", [])
            match    = set(verdict.get("flagged_issues", [])) == set(gt)
            cost_usd = result["cost"]["estimated_cost_usd"]

            case_results.append({
                "case_id":             case["case_id"],
                "domain":              case.get("domain", "?"),
                "ground_truth_issues": gt,
                "prediction":          case.get("prediction", "?"),
                "verdict":             verdict,
                "match":               match,
                "cost":                result["cost"],
                "transcript_path":     str(tx_path),
                "stop_reason":         result["stop_reason"],
                "speaker_selection":   result["speaker_selection"],
                "raw_verdict_message": result["raw_verdict_message"],
            })
            verdicts.append(verdict)
            gt_lists.append(gt)

            table_rows.append({
                "case_id":      str(case["case_id"]),
                "ground_truth": _format_issues(gt),
                "verdict":      _format_verdict(verdict),
                "match":        "yes" if match else "no",
                "cost":         f"${cost_usd:.6f}",
            })
            print(f"  → {_format_verdict(verdict)}")

        except Exception as exc:
            failures += 1
            err = f"{type(exc).__name__}: {exc}"
            case_results.append({
                "case_id": case["case_id"],
                "error":   err,
                "verdict": None,
                "match":   False,
            })
            table_rows.append({
                "case_id":      str(case["case_id"]),
                "ground_truth": _format_issues(case.get("ground_truth_issues", [])),
                "verdict":      "ERROR",
                "match":        "no",
                "cost":         "n/a",
            })
            print(f"  ✗ FAILED: {err}")

    # Summary metrics
    summary = compute_agreement(verdicts, gt_lists)
    summary["failures"] = failures
    summary["successful_cases"] = len(verdicts)

    # Save results
    payload = {
        "mode":       mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "provider":          llm_config.provider,
            "model":             llm_config.model,
            "temperature":       llm_config.temperature,
            "max_tokens":        llm_config.max_tokens,
            "speaker_selection": "single_llm" if args.single_llm else args.speaker_selection,
        },
        "summary": summary,
        "results": case_results,
    }

    results_path = run_dir / f"{mode}_results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also save a "latest" symlink-style copy
    latest_path = OUTPUT_ROOT / f"{slug}_{mode}_latest.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Print summary table
    print(f"\n{'═'*80}")
    _print_table(table_rows)
    print()
    print(f"Detection rate:    {summary['detection_rate']:.1f}%")
    print(f"False positive:    {summary['false_positive_rate']:.1f}%")
    print(f"Exact match:       {summary['exact_match_rate']:.1f}%")
    print(f"Successful:        {summary['successful_cases']}/{len(cases)}")
    print(f"Results:           {results_path}")
    print(f"Latest:            {latest_path}")
    print(f"Transcripts:       {tx_dir}")


if __name__ == "__main__":
    main()
