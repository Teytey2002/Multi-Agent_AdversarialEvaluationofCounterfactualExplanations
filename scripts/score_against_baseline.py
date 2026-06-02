"""Score LLM evaluators against the metrics-only reference system."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_FILE = PROJECT_ROOT / "results" / "metrics_only_outputs" / "metrics_only_latest.json"
DEFAULT_DEBATE_OUTPUTS_DIR = PROJECT_ROOT / "results" / "debate_outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "substitution_outputs"

SYSTEM_PATTERNS = {
    "single_llm": "*_single_llm_latest.json",
    "multi_agent": "*_multi_agent_latest.json",
}

SYSTEM_LABELS = {
    "single_llm": "Single-LLM",
    "multi_agent": "Multi-Agent",
}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _latest_mode_file(mode: str) -> Path | None:
    matches = sorted(DEFAULT_DEBATE_OUTPUTS_DIR.glob(SYSTEM_PATTERNS[mode]))
    if not matches:
        return None
    return matches[-1]


def _resolve_system_file(mode: str, explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    path = _latest_mode_file(mode)
    if path is None:
        print(
            f"Warning: no {SYSTEM_LABELS[mode]} latest file found; skipping this system.",
            file=sys.stderr,
        )
    return path


def _verdict_by_case(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    by_case: dict[int, dict[str, Any]] = {}
    for result in payload.get("results", []):
        verdict = result.get("verdict") or {}
        case_id = verdict.get("case_id", result.get("case_id"))
        if case_id is None:
            continue
        by_case[int(case_id)] = verdict
    return by_case


def _issue_set(verdict: dict[str, Any]) -> set[str]:
    return {str(issue) for issue in verdict.get("flagged_issues", [])}


def _field_match(candidate: dict[str, Any], reference: dict[str, Any], field: str) -> bool:
    return candidate.get(field) == reference.get(field)


def _score_system(
    system_verdicts: dict[int, dict[str, Any]],
    reference_verdicts: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    # Label-level confusion counts, treating each reference issue label as the
    # positive class. These feed the unified Phase-1 vocabulary (precision /
    # recall / F1) so substitution scoring and ground-truth scoring report the
    # same metrics. recall is the former "detection rate"; precision penalises
    # extra labels the reference did not flag.
    true_positives = 0   # labels in both reference and system
    false_negatives = 0  # reference labels the system missed
    false_positives = 0  # system labels absent from the reference
    exact_match_cases = 0
    assessment_matches = 0
    severity_matches = 0
    action_matches = 0
    per_case: dict[int, dict[str, Any]] = {}

    for case_id in sorted(reference_verdicts):
        reference_verdict = reference_verdicts[case_id]
        system_verdict = system_verdicts.get(case_id, {})

        reference_issues = _issue_set(reference_verdict)
        system_issues = _issue_set(system_verdict)
        missed = sorted(reference_issues - system_issues)
        extra = sorted(system_issues - reference_issues)
        exact = system_issues == reference_issues

        true_positives += len(reference_issues & system_issues)
        false_negatives += len(missed)
        false_positives += len(extra)
        if exact:
            exact_match_cases += 1
        if _field_match(system_verdict, reference_verdict, "overall_assessment"):
            assessment_matches += 1
        if _field_match(system_verdict, reference_verdict, "severity"):
            severity_matches += 1
        if _field_match(system_verdict, reference_verdict, "recommended_action"):
            action_matches += 1

        per_case[case_id] = {
            "flagged_issues": sorted(system_issues),
            "missed_issues": missed,
            "extra_issues": extra,
            "exact_match": exact,
        }

    total_cases = len(reference_verdicts)
    predicted_positives = true_positives + false_positives
    reference_positives = true_positives + false_negatives

    recall = true_positives / reference_positives * 100 if reference_positives else 100.0
    precision = true_positives / predicted_positives * 100 if predicted_positives else 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "exact_match_rate": round(exact_match_cases / total_cases * 100, 2) if total_cases else 0.0,
        "assessment_agreement": round(assessment_matches / total_cases * 100, 2) if total_cases else 0.0,
        "severity_agreement": round(severity_matches / total_cases * 100, 2) if total_cases else 0.0,
        "recommended_action_agreement": round(action_matches / total_cases * 100, 2) if total_cases else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_cases": total_cases,
        "cases_with_perfect_match": exact_match_cases,
    }
    return summary, per_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score LLM evaluators against the metrics-only reference system."
    )
    parser.add_argument(
        "--reference-file",
        default=str(DEFAULT_REFERENCE_FILE),
        help="Path to metrics-only reference results JSON.",
    )
    parser.add_argument(
        "--single-llm-file",
        default=None,
        help="Path to single-LLM latest results JSON. Defaults to discovery under results/debate_outputs/.",
    )
    parser.add_argument(
        "--multi-agent-file",
        default=None,
        help="Path to multi-agent latest results JSON. Defaults to discovery under results/debate_outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for substitution score outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_file = Path(args.reference_file)
    if not reference_file.is_absolute():
        reference_file = PROJECT_ROOT / reference_file

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    system_files = {
        "single_llm": _resolve_system_file("single_llm", args.single_llm_file),
        "multi_agent": _resolve_system_file("multi_agent", args.multi_agent_file),
    }
    system_files = {mode: path for mode, path in system_files.items() if path is not None}

    reference_payload = _load_json(reference_file)
    reference_verdicts = _verdict_by_case(reference_payload)
    if not reference_verdicts:
        raise ValueError(f"No reference verdicts found in {reference_file}")

    summaries: dict[str, dict[str, Any]] = {}
    per_system_case: dict[str, dict[int, dict[str, Any]]] = {}
    systems_evaluated: dict[str, str] = {}

    for mode, path in system_files.items():
        if not path.exists():
            print(
                f"Warning: {SYSTEM_LABELS[mode]} file does not exist: {_display_path(path)}; skipping.",
                file=sys.stderr,
            )
            continue
        payload = _load_json(path)
        system_verdicts = _verdict_by_case(payload)
        summary, case_rows = _score_system(system_verdicts, reference_verdicts)
        summaries[mode] = summary
        per_system_case[mode] = case_rows
        systems_evaluated[mode] = _display_path(path)

    per_case: list[dict[str, Any]] = []
    for case_id in sorted(reference_verdicts):
        reference_issues = sorted(_issue_set(reference_verdicts[case_id]))
        row: dict[str, Any] = {
            "case_id": case_id,
            "reference_flagged_issues": reference_issues,
        }
        for mode in ("single_llm", "multi_agent"):
            if mode in per_system_case:
                row[mode] = per_system_case[mode][case_id]
        per_case.append(row)

    created_at = datetime.now().replace(microsecond=0).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "mode": "substitution_scoring",
        "created_at": created_at,
        "reference": _display_path(reference_file),
        "systems_evaluated": systems_evaluated,
        "summary": summaries,
        "per_case": per_case,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamped_path = output_dir / f"substitution_scores_{timestamp}.json"
    latest_path = output_dir / "substitution_scores_latest.json"
    for path in (timestamped_path, latest_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")

    print("System        Prec%  Recall%   F1%   ExactMatch%   AssessAgree%")
    for mode in ("single_llm", "multi_agent"):
        if mode not in summaries:
            continue
        summary = summaries[mode]
        print(
            f"{SYSTEM_LABELS[mode]:<12}"
            f"{summary['precision']:>6.1f}%"
            f"{summary['recall']:>8.1f}%"
            f"{summary['f1']:>6.1f}%"
            f"{summary['exact_match_rate']:>12.1f}%"
            f"{summary['assessment_agreement']:>13.1f}%"
        )
    print(f"Results: {_display_path(timestamped_path)}")
    print(f"Latest:  {_display_path(latest_path)}")


if __name__ == "__main__":
    main()
