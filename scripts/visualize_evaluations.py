"""Render SVG dashboards for scored evaluation outputs.

The script is dependency-free on purpose: it reads the existing JSON outputs
from metrics-only, single-LLM, and multi-agent runs, then writes static SVG
figures that can be committed with reports or regenerated locally.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_ONLY = PROJECT_ROOT / "results" / "metrics_only_outputs" / "metrics_only_latest.json"
DEFAULT_SINGLE_LLM = (
    PROJECT_ROOT / "results" / "debate_outputs" / "llama-3.1-8b-instant_single_llm_latest.json"
)
DEFAULT_MULTI_AGENT = (
    PROJECT_ROOT / "results" / "debate_outputs" / "llama-3.1-8b-instant_multi_agent_latest.json"
)
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"

ISSUE_ORDER = [
    "fragile_counterfactual",
    "implausible_time_dependent_change",
    "inconsistent_work_profile",
    "too_many_changes",
    "unactionable_capital_shift",
    "extreme_working_hours",
]

COLORS = {
    "ink": "#172033",
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
    "purple_bg": "#ede9fe",
}


@dataclass(frozen=True)
class EvaluationView:
    name: str
    payload: dict[str, Any]
    precision: float
    recall: float
    f1: float
    exact_match: float
    match_count: int
    predicted_total: int
    caught_total: int
    ground_truth_total: int
    total_cases: int
    total_cost: float
    ground_truth_counts: Counter[str]
    prediction_counts: Counter[str]
    missed_counts: Counter[str]
    extra_counts: Counter[str]
    per_case: dict[int, dict[str, Any]]


def _format_issue(issue: str) -> str:
    return issue.replace("_", " ")


def _percent(value: float) -> str:
    return f"{value:.1f}%"


def _svg_text(
    x: int,
    y: int,
    text: str,
    *,
    size: int = 14,
    weight: str = "400",
    color: str | None = None,
) -> str:
    fill = color or COLORS["ink"]
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(text)}</text>'
    )


def _panel(x: int, y: int, w: int, h: int) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}"/>'


def _metric_card(x: int, y: int, w: int, h: int, title: str, value: str, subtitle: str, color: str) -> str:
    return "\n".join(
        [
            _panel(x, y, w, h),
            _svg_text(x + 18, y + 28, title, size=13, color=COLORS["muted"]),
            _svg_text(x + 18, y + 64, value, size=28, weight="700", color=color),
            _svg_text(x + 18, y + 92, subtitle, size=12, color=COLORS["muted"]),
        ]
    )


def _count_issues(results: list[dict[str, Any]], key_path: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        value: Any = result
        for key in key_path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, list):
            counts.update(str(issue) for issue in value)
    return counts


def _total_cost(results: list[dict[str, Any]]) -> float:
    total = 0.0
    for result in results:
        cost = result.get("cost", {})
        if isinstance(cost, dict):
            total += float(cost.get("estimated_cost_usd") or 0.0)
    return total


def _build_view(name: str, payload: dict[str, Any]) -> EvaluationView:
    summary = payload["summary"]
    results = payload["results"]
    ground_truth_counts = _count_issues(results, ("ground_truth_issues",))
    prediction_counts = _count_issues(results, ("verdict", "flagged_issues"))
    predicted_total = sum(prediction_counts.values())
    caught_total = int(summary["caught_issues"])
    ground_truth_total = int(summary["total_ground_truth_issues"])
    precision = caught_total / predicted_total * 100 if predicted_total else 0.0
    recall = float(summary["detection_rate"])
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    match_count = sum(1 for result in results if result.get("match"))

    missed_counts: Counter[str] = Counter()
    extra_counts: Counter[str] = Counter()
    per_case: dict[int, dict[str, Any]] = {}
    for item in summary.get("per_case", []):
        case_id = int(item["case_id"])
        per_case[case_id] = item
        missed_counts.update(str(issue) for issue in item.get("missed_issues", []))
        extra_counts.update(str(issue) for issue in item.get("extra_issues", []))

    return EvaluationView(
        name=name,
        payload=payload,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=float(summary["exact_match_rate"]),
        match_count=match_count,
        predicted_total=predicted_total,
        caught_total=caught_total,
        ground_truth_total=ground_truth_total,
        total_cases=int(summary["total_cases"]),
        total_cost=_total_cost(results),
        ground_truth_counts=ground_truth_counts,
        prediction_counts=prediction_counts,
        missed_counts=missed_counts,
        extra_counts=extra_counts,
        per_case=per_case,
    )


def _case_status(case: dict[str, Any]) -> str:
    if case.get("match"):
        return "match"
    missed = bool(case.get("missed_issues"))
    extra = bool(case.get("extra_issues"))
    if missed and extra:
        return "mixed"
    if missed:
        return "missed"
    if extra:
        return "extra"
    return "different"


def _status_colors(status: str) -> tuple[str, str, str]:
    if status == "match":
        return COLORS["green_bg"], COLORS["green"], "match"
    if status == "missed":
        return COLORS["red_bg"], COLORS["red"], "missed"
    if status == "extra":
        return COLORS["amber_bg"], COLORS["amber"], "extra"
    if status == "mixed":
        return COLORS["purple_bg"], COLORS["purple"], "mixed"
    return "#eef2ff", COLORS["blue"], "diff"


def _top_counts(counts: Counter[str], *, empty: str) -> str:
    if not counts:
        return empty
    parts = [f"{_format_issue(issue)} ({count})" for issue, count in counts.most_common(3)]
    return ", ".join(parts)


def _bar_chart(view: EvaluationView, *, x: int, y: int, w: int = 620, h: int = 282) -> str:
    max_count = max([1, *view.ground_truth_counts.values(), *view.prediction_counts.values()])
    label_w = 230
    column_gap = 170
    max_bar = 145
    parts = [
        _panel(x, y, w, h),
        _svg_text(x + 18, y + 31, "Issue Coverage By Label", size=17, weight="700"),
        _svg_text(x + label_w, y + 58, "Ground truth", size=12, color=COLORS["green"]),
        _svg_text(x + label_w + column_gap, y + 58, view.name, size=12, color=COLORS["blue"]),
    ]

    start_y = y + 84
    for idx, issue in enumerate(ISSUE_ORDER):
        row_y = start_y + idx * 34
        gt = view.ground_truth_counts.get(issue, 0)
        pred = view.prediction_counts.get(issue, 0)
        gt_len = int((gt / max_count) * max_bar)
        pred_len = int((pred / max_count) * max_bar)
        parts.extend(
            [
                _svg_text(x + 18, row_y + 13, _format_issue(issue), size=12),
                f'<rect x="{x + label_w}" y="{row_y}" width="{gt_len if gt else 0}" height="12" rx="3" fill="{COLORS["green"]}"/>',
                f'<rect x="{x + label_w + column_gap}" y="{row_y}" width="{pred_len if pred else 0}" height="12" rx="3" fill="{COLORS["blue"]}"/>',
                _svg_text(x + label_w + max(gt_len, 2) + 8, row_y + 11, str(gt), size=11, color=COLORS["muted"]),
                _svg_text(x + label_w + column_gap + max(pred_len, 2) + 8, row_y + 11, str(pred), size=11, color=COLORS["muted"]),
            ]
        )

    return "\n".join(parts)


def _case_grid(view: EvaluationView, *, x: int, y: int, w: int = 500, h: int = 282) -> str:
    parts = [
        _panel(x, y, w, h),
        _svg_text(x + 18, y + 31, "Per-Case Agreement", size=17, weight="700"),
    ]
    legend = [
        ("match", COLORS["green_bg"], COLORS["green"]),
        ("missed", COLORS["red_bg"], COLORS["red"]),
        ("extra", COLORS["amber_bg"], COLORS["amber"]),
        ("mixed", COLORS["purple_bg"], COLORS["purple"]),
    ]
    lx = x + 18
    for label, bg, fg in legend:
        parts.append(f'<rect x="{lx}" y="{y + 49}" width="14" height="14" rx="3" fill="{bg}" stroke="{fg}"/>')
        parts.append(_svg_text(lx + 20, y + 61, label, size=11, color=COLORS["muted"]))
        lx += 104

    cell_w = 82
    cell_h = 58
    start_x = x + 22
    start_y = y + 84
    case_ids = sorted(view.per_case)
    for idx, case_id in enumerate(case_ids):
        item = view.per_case[case_id]
        status = _case_status(item)
        bg, fg, label = _status_colors(status)
        row = idx // 5
        col = idx % 5
        cx = start_x + col * (cell_w + 10)
        cy = start_y + row * (cell_h + 18)
        parts.extend(
            [
                f'<rect x="{cx}" y="{cy}" width="{cell_w}" height="{cell_h}" rx="7" fill="{bg}" stroke="{fg}"/>',
                _svg_text(cx + 12, cy + 23, f"Case {case_id}", size=14, weight="700", color=fg),
                _svg_text(cx + 12, cy + 45, label, size=12, color=fg),
            ]
        )

    exact_cases = [str(case_id) for case_id in case_ids if view.per_case[case_id].get("match")]
    exact_text = ", ".join(exact_cases) if exact_cases else "none"
    parts.append(_svg_text(x + 18, y + h - 25, f"Exact-match cases: {exact_text}.", size=12, color=COLORS["muted"]))
    return "\n".join(parts)


def build_single_svg(name: str, payload: dict[str, Any]) -> str:
    view = _build_view(name, payload)
    created_at = payload.get("created_at", "unknown")
    missed_text = _top_counts(view.missed_counts, empty="none")
    extra_text = _top_counts(view.extra_counts, empty="none")
    cost_text = "no LLM cost" if view.total_cost == 0 else f"${view.total_cost:.6f} estimated LLM cost"

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850">',
        f'<rect width="1200" height="850" fill="{COLORS["bg"]}"/>',
        _svg_text(42, 54, f"{view.name} Evaluation", size=30, weight="700"),
        _svg_text(42, 82, f"Run: {created_at} | Compared against draft human-perspective labels", size=14, color=COLORS["muted"]),
        _metric_card(42, 112, 250, 110, "Issue Recall", _percent(view.recall), f"{view.caught_total} / {view.ground_truth_total} labels caught", COLORS["green"]),
        _metric_card(318, 112, 250, 110, "Issue Precision", _percent(view.precision), f"{view.predicted_total - view.caught_total} extra issue(s)", COLORS["blue"]),
        _metric_card(594, 112, 250, 110, "Issue F1", _percent(view.f1), "Balance of precision and recall", COLORS["purple"]),
        _metric_card(870, 112, 250, 110, "Exact Match", _percent(view.exact_match), f"{view.match_count} / {view.total_cases} cases", COLORS["amber"]),
        _bar_chart(view, x=42, y=254),
        _case_grid(view, x=676, y=254),
        _panel(42, 560, 1078, 226),
        _svg_text(64, 594, "Interpretation", size=18, weight="700"),
        _svg_text(64, 626, f"{view.name} catches {view.caught_total} of {view.ground_truth_total} reference issues and exactly matches {view.match_count} of {view.total_cases} cases.", size=14),
        _svg_text(64, 654, f"Most missed labels: {missed_text}.", size=14, color=COLORS["red"]),
        _svg_text(64, 682, f"Most extra labels: {extra_text}.", size=14, color=COLORS["amber"]),
        _svg_text(64, 722, f"Cost marker: {cost_text}.", size=15, weight="700"),
        _svg_text(64, 754, "Use this dashboard to see whether the system improves semantic judgment without over-flagging issues.", size=13, color=COLORS["muted"]),
        "</svg>",
    ]
    return "\n".join(parts)


def _metric_bar_group(views: list[EvaluationView], *, x: int, y: int, title: str, metric: str) -> str:
    max_w = 260
    row_h = 34
    parts = [_svg_text(x, y, title, size=15, weight="700")]
    for idx, view in enumerate(views):
        value = float(getattr(view, metric))
        row_y = y + 24 + idx * row_h
        width = int(max_w * value / 100)
        parts.extend(
            [
                _svg_text(x, row_y + 12, view.name, size=12, color=COLORS["muted"]),
                f'<rect x="{x + 120}" y="{row_y}" width="{max_w}" height="14" rx="4" fill="#eef2f6"/>',
                f'<rect x="{x + 120}" y="{row_y}" width="{width}" height="14" rx="4" fill="{COLORS["blue"]}"/>',
                _svg_text(x + 120 + max_w + 12, row_y + 13, _percent(value), size=12, weight="700"),
            ]
        )
    return "\n".join(parts)


def _comparison_case_matrix(views: list[EvaluationView], *, x: int, y: int) -> str:
    parts = [
        _panel(x, y, 740, 252),
        _svg_text(x + 18, y + 32, "Case-Level Agreement Matrix", size=18, weight="700"),
    ]
    case_ids = sorted({case_id for view in views for case_id in view.per_case})
    for idx, case_id in enumerate(case_ids):
        parts.append(_svg_text(x + 170 + idx * 48, y + 64, str(case_id), size=12, weight="700", color=COLORS["muted"]))

    for row_idx, view in enumerate(views):
        row_y = y + 84 + row_idx * 46
        parts.append(_svg_text(x + 18, row_y + 25, view.name, size=13, weight="700"))
        for col_idx, case_id in enumerate(case_ids):
            case = view.per_case.get(case_id, {"match": False, "missed_issues": [], "extra_issues": []})
            bg, fg, label = _status_colors(_case_status(case))
            cx = x + 158 + col_idx * 48
            parts.append(f'<rect x="{cx}" y="{row_y}" width="34" height="34" rx="6" fill="{bg}" stroke="{fg}"/>')
            parts.append(_svg_text(cx + 9, row_y + 23, label[0].upper(), size=13, weight="700", color=fg))

    legend_y = y + 220
    legend = [
        ("M match", COLORS["green_bg"], COLORS["green"]),
        ("X missed", COLORS["red_bg"], COLORS["red"]),
        ("E extra", COLORS["amber_bg"], COLORS["amber"]),
        ("B both", COLORS["purple_bg"], COLORS["purple"]),
    ]
    lx = x + 18
    for label, bg, fg in legend:
        parts.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" rx="3" fill="{bg}" stroke="{fg}"/>')
        parts.append(_svg_text(lx + 20, legend_y + 12, label, size=11, color=COLORS["muted"]))
        lx += 104
    return "\n".join(parts)


def _comparison_issue_table(views: list[EvaluationView], *, x: int, y: int) -> str:
    parts = [
        _panel(x, y, 500, 252),
        _svg_text(x + 18, y + 32, "Issue Divergence", size=18, weight="700"),
    ]
    for view_idx, view in enumerate(views[:3]):
        base_x = x + 205 + view_idx * 92
        parts.append(_svg_text(base_x - 8, y + 48, view.name, size=10, color=COLORS["muted"]))
        parts.append(_svg_text(base_x, y + 64, "miss", size=10, weight="700", color=COLORS["red"]))
        parts.append(_svg_text(base_x + 42, y + 64, "extra", size=10, weight="700", color=COLORS["amber"]))

    start_y = y + 86
    for idx, issue in enumerate(ISSUE_ORDER):
        row_y = start_y + idx * 26
        parts.append(_svg_text(x + 18, row_y + 12, _format_issue(issue), size=11))
        for view_idx, view in enumerate(views[:3]):
            base_x = x + 205 + view_idx * 92
            miss = view.missed_counts.get(issue, 0)
            extra = view.extra_counts.get(issue, 0)
            parts.append(_svg_text(base_x, row_y + 12, str(miss), size=12, weight="700", color=COLORS["red"] if miss else COLORS["muted"]))
            parts.append(_svg_text(base_x + 45, row_y + 12, str(extra), size=12, weight="700", color=COLORS["amber"] if extra else COLORS["muted"]))
    return "\n".join(parts)


def build_comparison_svg(named_payloads: list[tuple[str, dict[str, Any]]]) -> str:
    views = [_build_view(name, payload) for name, payload in named_payloads]
    best_f1 = max(views, key=lambda view: view.f1)
    best_exact = max(views, key=lambda view: view.exact_match)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="900" viewBox="0 0 1300 900">',
        f'<rect width="1300" height="900" fill="{COLORS["bg"]}"/>',
        _svg_text(42, 54, "Evaluation System Comparison", size=30, weight="700"),
        _svg_text(42, 82, "Metrics-only, single-LLM, and multi-agent outputs scored against the same draft reference labels.", size=14, color=COLORS["muted"]),
        _panel(42, 112, 1216, 236),
        _metric_bar_group(views, x=66, y=148, title="Precision", metric="precision"),
        _metric_bar_group(views, x=674, y=148, title="Recall", metric="recall"),
        _metric_bar_group(views, x=66, y=260, title="F1", metric="f1"),
        _metric_bar_group(views, x=674, y=260, title="Exact Match", metric="exact_match"),
        _comparison_case_matrix(views, x=42, y=382),
        _comparison_issue_table(views, x=758, y=382),
        _panel(42, 668, 1216, 164),
        _svg_text(64, 704, "Reading The Result", size=18, weight="700"),
        _svg_text(64, 736, f"Best F1: {best_f1.name} at {_percent(best_f1.f1)}. Best exact-match rate: {best_exact.name} at {_percent(best_exact.exact_match)}.", size=14),
        _svg_text(64, 764, "The comparison separates threshold-like detection strength from semantic calibration errors such as missing or over-adding issue labels.", size=14),
        _svg_text(64, 792, "Use the divergence table to decide where prompts, role instructions, or deterministic label rules need calibration before the final write-up.", size=14, color=COLORS["muted"]),
        "</svg>",
    ]
    return "\n".join(parts)


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_svg(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    print(f"Rendered {path}")


def _default_inputs() -> list[tuple[str, Path]]:
    return [
        ("Metrics-Only", DEFAULT_METRICS_ONLY),
        ("Single LLM", DEFAULT_SINGLE_LLM),
        ("Multi-Agent", DEFAULT_MULTI_AGENT),
    ]


def render_suite(output_dir: Path) -> None:
    named_payloads = [(name, _load_payload(path)) for name, path in _default_inputs()]
    for name, payload in named_payloads:
        filename = name.lower().replace(" ", "_").replace("-", "_") + "_summary.svg"
        _write_svg(output_dir / filename, build_single_svg(name, payload))
    _write_svg(output_dir / "system_comparison_summary.svg", build_comparison_svg(named_payloads))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SVG dashboards for evaluation JSON outputs.")
    subparsers = parser.add_subparsers(dest="command")

    suite = subparsers.add_parser("suite", help="Render the default metrics-only, single-LLM, multi-agent, and comparison figures.")
    suite.add_argument("--output-dir", default=str(DEFAULT_FIGURES_DIR), help=f"Output directory (default: {DEFAULT_FIGURES_DIR})")

    single = subparsers.add_parser("single", help="Render one evaluation dashboard.")
    single.add_argument("--input", required=True, help="Input evaluation JSON.")
    single.add_argument("--system-name", required=True, help="Display name for the evaluated system.")
    single.add_argument("--output", required=True, help="Output SVG path.")

    compare = subparsers.add_parser("compare", help="Render one comparison dashboard.")
    compare.add_argument("--inputs", nargs="+", required=True, help="Input evaluation JSON files.")
    compare.add_argument("--system-names", nargs="+", required=True, help="Display names matching --inputs order.")
    compare.add_argument("--output", required=True, help="Output SVG path.")

    parser.set_defaults(command="suite", output_dir=str(DEFAULT_FIGURES_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "single":
        payload = _load_payload(Path(args.input))
        _write_svg(Path(args.output), build_single_svg(args.system_name, payload))
        return

    if args.command == "compare":
        if len(args.inputs) != len(args.system_names):
            raise SystemExit("--inputs and --system-names must have the same length")
        named_payloads = [
            (name, _load_payload(Path(path)))
            for name, path in zip(args.system_names, args.inputs, strict=True)
        ]
        _write_svg(Path(args.output), build_comparison_svg(named_payloads))
        return

    render_suite(Path(args.output_dir))


if __name__ == "__main__":
    main()
