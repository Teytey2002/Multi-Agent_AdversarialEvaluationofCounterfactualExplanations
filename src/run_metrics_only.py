"""Run the deterministic metrics-only counterfactual evaluator.

The output schema mirrors the LLM Judge verdict schema so results can be
compared directly with single-LLM and multi-agent runs once reference labels
are available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.utils import compute_agreement
from evaluators.metrics_only import evaluate_case_metrics_only


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "results" / "cases.json"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "metrics_only_outputs"


def load_cases(path: Path, case_ids: list[int] | None = None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if case_ids is None:
        return cases

    id_set = set(case_ids)
    selected = [case for case in cases if case["case_id"] in id_set]
    if not selected:
        raise ValueError(f"No cases matched --case-ids {case_ids}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic metrics-only evaluation on case JSON.",
    )
    parser.add_argument(
        "--cases-file",
        type=str,
        default=str(CASES_PATH),
        help=f"Path to cases JSON (default: {CASES_PATH}).",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        help="Run only specific case IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT),
        help=f"Output directory (default: {OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.cases_file), case_ids=args.case_ids)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    run_dir = output_root / f"metrics_only_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    ground_truth: list[list[str]] = []

    for case in cases:
        verdict = evaluate_case_metrics_only(case)
        verdicts.append(verdict)
        gt = list(case.get("ground_truth_issues", []))
        ground_truth.append(gt)
        results.append({
            "case_id": case["case_id"],
            "ground_truth_issues": gt,
            "verdict": verdict,
            "match": set(verdict.get("flagged_issues", [])) == set(gt),
        })

    payload = {
        "mode": "metrics_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "cases_file": str(Path(args.cases_file)),
            "case_ids": args.case_ids,
            "evaluator": "evaluators.metrics_only.evaluate_case_metrics_only",
        },
        "summary": compute_agreement(verdicts, ground_truth),
        "results": results,
    }

    results_path = run_dir / "metrics_only_results.json"
    latest_path = output_root / "metrics_only_latest.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Mode: metrics_only")
    print(f"Cases: {len(cases)} (IDs: {[case['case_id'] for case in cases]})")
    for item in results:
        verdict = item["verdict"]
        issues = verdict.get("flagged_issues", [])
        issue_text = ", ".join(issues) if issues else "clean"
        print(
            f"  case {item['case_id']:>2}: "
            f"{verdict['overall_assessment']} / {verdict['severity']} "
            f"({issue_text})"
        )

    print(f"Results: {results_path}")
    print(f"Latest:  {latest_path}")


if __name__ == "__main__":
    main()
