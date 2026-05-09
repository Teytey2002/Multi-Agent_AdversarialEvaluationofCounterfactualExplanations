"""Create an SVG visual summary for metrics-only evaluation results.

The project environment does not require a plotting dependency for this script.
It renders a compact SVG dashboard directly from the metrics-only JSON output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "metrics_only_outputs" / "metrics_only_latest.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "metrics_only_outputs"
    / "visuals"
    / "metrics_only_summary.svg"
)

ISSUE_ORDER = [
    "fragile_counterfactual",
    "implausible_time_dependent_change",
    "inconsistent_work_profile",
    "too_many_changes",
    "unactionable_capital_shift",
    "extreme_working_hours",
]

COLORS = {
    "ink": "#18212f",
    "muted": "#667085",
    "grid": "#d0d5dd",
    "bg": "#f8fafc",
    "panel": "#ffffff",
    "green": "#2f855a",
    "green_bg": "#dcfce7",
    "blue": "#2563eb",
    "blue_bg": "#dbeafe",
    "red": "#c2410c",
    "red_bg": "#ffedd5",
    "amber": "#b45309",
    "amber_bg": "#fef3c7",
    "purple": "#6d28d9",
}


def _count_issues(results: list[dict[str, Any]], key_path: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        value: Any = result
        for key in key_path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        for issue in value if isinstance(value, list) else []:
            counts[str(issue)] += 1
    return counts


def _case_status(result: dict[str, Any], per_case: dict[int, dict[str, Any]]) -> str:
    case_id = int(result["case_id"])
    details = per_case.get(case_id, {})
    if result.get("match"):
        return "match"
    if details.get("missed_issues"):
        return "missed"
    if details.get("extra_issues"):
        return "extra"
    return "different"


def _percent(value: float) -> str:
    return f"{value:.1f}%"


def _svg_text(x: int, y: int, text: str, *, size: int = 14, weight: str = "400", color: str | None = None) -> str:
    fill = color or COLORS["ink"]
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(text)}</text>'
    )


def _card(x: int, y: int, w: int, h: int, title: str, value: str, subtitle: str, color: str) -> str:
    return "\n".join([
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}"/>',
        _svg_text(x + 18, y + 28, title, size=13, color=COLORS["muted"]),
        _svg_text(x + 18, y + 63, value, size=28, weight="700", color=color),
        _svg_text(x + 18, y + 91, subtitle, size=12, color=COLORS["muted"]),
    ])


def _bar_chart(
    x: int,
    y: int,
    title: str,
    gt_counts: Counter[str],
    pred_counts: Counter[str],
) -> str:
    max_count = max([1, *gt_counts.values(), *pred_counts.values()])
    chart_w = 620
    row_h = 34
    label_w = 230
    bar_w = chart_w - label_w - 70
    parts = [
        f'<rect x="{x}" y="{y}" width="{chart_w}" height="278" rx="8" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}"/>',
        _svg_text(x + 18, y + 30, title, size=17, weight="700"),
        _svg_text(x + label_w, y + 56, "Ground truth", size=12, color=COLORS["green"]),
        _svg_text(x + label_w + 170, y + 56, "Metrics-only", size=12, color=COLORS["blue"]),
    ]

    start_y = y + 82
    for idx, issue in enumerate(ISSUE_ORDER):
        row_y = start_y + idx * row_h
        gt = gt_counts.get(issue, 0)
        pred = pred_counts.get(issue, 0)
        gt_len = int((gt / max_count) * (bar_w * 0.45))
        pred_len = int((pred / max_count) * (bar_w * 0.45))
        label = issue.replace("_", " ")
        parts.extend([
            _svg_text(x + 18, row_y + 13, label, size=12, color=COLORS["ink"]),
            f'<rect x="{x + label_w}" y="{row_y}" width="{max(gt_len, 2) if gt else 0}" height="12" rx="3" fill="{COLORS["green"]}"/>',
            f'<rect x="{x + label_w + 170}" y="{row_y}" width="{max(pred_len, 2) if pred else 0}" height="12" rx="3" fill="{COLORS["blue"]}"/>',
            _svg_text(x + label_w + max(gt_len, 2) + 8, row_y + 11, str(gt), size=11, color=COLORS["muted"]),
            _svg_text(x + label_w + 170 + max(pred_len, 2) + 8, row_y + 11, str(pred), size=11, color=COLORS["muted"]),
        ])

    parts.append(_svg_text(x + 18, y + 260, "Main pattern: the baseline misses inconsistent work profile and adds one extreme-hours issue.", size=12, color=COLORS["muted"]))
    return "\n".join(parts)


def _case_grid(x: int, y: int, results: list[dict[str, Any]], per_case: dict[int, dict[str, Any]]) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="500" height="278" rx="8" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}"/>',
        _svg_text(x + 18, y + 30, "Per-Case Agreement", size=17, weight="700"),
    ]
    legend = [("match", COLORS["green_bg"], COLORS["green"]), ("missed issue", COLORS["red_bg"], COLORS["red"]), ("extra issue", COLORS["amber_bg"], COLORS["amber"])]
    lx = x + 18
    for label, bg, fg in legend:
        parts.append(f'<rect x="{lx}" y="{y + 48}" width="14" height="14" rx="3" fill="{bg}" stroke="{fg}"/>')
        parts.append(_svg_text(lx + 20, y + 60, label, size=11, color=COLORS["muted"]))
        lx += 120

    cell_w = 82
    cell_h = 58
    start_x = x + 22
    start_y = y + 82
    for idx, result in enumerate(results):
        case_id = int(result["case_id"])
        status = _case_status(result, per_case)
        row = idx // 5
        col = idx % 5
        cx = start_x + col * (cell_w + 10)
        cy = start_y + row * (cell_h + 18)
        if status == "match":
            bg, fg = COLORS["green_bg"], COLORS["green"]
            status_text = "match"
        elif status == "missed":
            bg, fg = COLORS["red_bg"], COLORS["red"]
            status_text = "missed"
        elif status == "extra":
            bg, fg = COLORS["amber_bg"], COLORS["amber"]
            status_text = "extra"
        else:
            bg, fg = "#eef2ff", COLORS["purple"]
            status_text = "diff"
        parts.extend([
            f'<rect x="{cx}" y="{cy}" width="{cell_w}" height="{cell_h}" rx="7" fill="{bg}" stroke="{fg}"/>',
            _svg_text(cx + 12, cy + 23, f"Case {case_id}", size=14, weight="700", color=fg),
            _svg_text(cx + 12, cy + 45, status_text, size=12, color=fg),
        ])

    parts.append(_svg_text(x + 18, y + 252, "Exact-match cases: 1, 3, 5, 7, 8, 9.", size=12, color=COLORS["muted"]))
    return "\n".join(parts)


def build_svg(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    results = payload["results"]
    per_case = {int(item["case_id"]): item for item in summary.get("per_case", [])}

    gt_counts = _count_issues(results, ("ground_truth_issues",))
    pred_counts = _count_issues(results, ("verdict", "flagged_issues"))
    predicted_total = sum(pred_counts.values())
    caught = float(summary["caught_issues"])
    precision = caught / predicted_total * 100 if predicted_total else 0.0
    recall = float(summary["detection_rate"])
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    match_count = sum(1 for result in results if result.get("match"))
    missed_cases = [item for item in summary.get("per_case", []) if item.get("missed_issues")]
    extra_cases = [item for item in summary.get("per_case", []) if item.get("extra_issues")]

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850">',
        f'<rect width="1200" height="850" fill="{COLORS["bg"]}"/>',
        _svg_text(42, 54, "Metrics-Only Baseline Evaluation", size=30, weight="700"),
        _svg_text(42, 82, f"Run: {payload.get('created_at', 'unknown')} | Compared against draft human-perspective labels", size=14, color=COLORS["muted"]),
        _card(42, 112, 250, 110, "Issue Recall", _percent(recall), f'{int(summary["caught_issues"])} / {int(summary["total_ground_truth_issues"])} labels caught', COLORS["green"]),
        _card(318, 112, 250, 110, "Issue Precision", _percent(precision), f"{predicted_total - int(caught)} extra issue(s)", COLORS["blue"]),
        _card(594, 112, 250, 110, "Issue F1", _percent(f1), "Balance of precision and recall", COLORS["purple"]),
        _card(870, 112, 250, 110, "Exact Match", _percent(float(summary["exact_match_rate"])), f"{match_count} / {int(summary['total_cases'])} cases", COLORS["amber"]),
        _bar_chart(42, 254, "Issue Coverage By Label", gt_counts, pred_counts),
        _case_grid(676, 254, results, per_case),
        f'<rect x="42" y="560" width="1078" height="226" rx="8" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}"/>',
        _svg_text(64, 594, "Interpretation", size=18, weight="700"),
        _svg_text(64, 626, f"The deterministic baseline catches {int(caught)} of {int(summary['total_ground_truth_issues'])} draft reference issues and exactly matches {match_count} of {int(summary['total_cases'])} cases.", size=14),
        _svg_text(64, 654, "Where it misses: inconsistent_work_profile in cases " + ", ".join(str(i["case_id"]) for i in missed_cases if "inconsistent_work_profile" in i.get("missed_issues", [])) + ".", size=14, color=COLORS["red"]),
        _svg_text(64, 682, "Where it over-flags: extreme_working_hours in case " + ", ".join(str(i["case_id"]) for i in extra_cases) + ".", size=14, color=COLORS["amber"]),
        _svg_text(64, 722, "Takeaway: metrics-only is strong on threshold-based issues, but weaker on semantic work-profile judgment.", size=15, weight="700"),
        _svg_text(64, 754, "This creates a useful target for the later single-LLM and multi-agent stages: improve semantic calibration without inventing extra issues.", size=13, color=COLORS["muted"]),
        "</svg>",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an SVG dashboard from metrics-only results.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input metrics-only JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output SVG path (default: {DEFAULT_OUTPUT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(payload), encoding="utf-8")

    print(f"Rendered metrics-only visual summary -> {output_path}")


if __name__ == "__main__":
    main()
