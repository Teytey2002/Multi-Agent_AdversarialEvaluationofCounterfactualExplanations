# Project Structure

This repository is organized by responsibility:

```text
repo/
|-- annotations/
|   `-- ground_truth_labels.json
|-- docs/
|   |-- architecture/
|   |-- methodology/
|   |-- reports/
|   |-- references/
|   `-- presentation/
|-- models/
|-- results/
|-- scripts/
|   |-- run_metrics_only.py
|   |-- run_debate.py
|   `-- visualize_metrics_only.py
|-- src/
|   |-- agents/
|   |-- evaluators/
|   |-- pipeline/
|   `-- policy/
`-- tests/
```

## Responsibilities

| Folder | Responsibility |
|---|---|
| `src/pipeline/` | ML pipeline stages and helpers: data loading, preprocessing, training, prediction, counterfactual generation, metrics, and case building. |
| `src/policy/` | Recourse feature policy and deterministic heuristics used to evaluate counterfactual plausibility. |
| `src/evaluators/` | Non-LLM evaluators, currently the deterministic metrics-only baseline. |
| `src/agents/` | AutoGen single-LLM and multi-agent debate implementation. |
| `scripts/` | Thin runnable entry points for evaluation and visualization. |
| `annotations/` | Human/team reference labels and label schemas. |
| `results/` | Generated pipeline and experiment artifacts. Timestamped LLM runs remain under ignored result-output folders. |
| `docs/methodology/` | Methodological explanations and rationale documents. |
| `docs/reports/` | Experiment comparison reports and current result summaries. |
| `docs/architecture/` | Project structure and script reference documentation. |
| `docs/references/` | External papers or reference PDFs. |
| `docs/presentation/` | Presentation material. |

## Main Commands

Run commands from the repository root with `PYTHONPATH=src`.

```powershell
$env:PYTHONPATH="src"; python -m pipeline.explore_data
$env:PYTHONPATH="src"; python -m pipeline.train
$env:PYTHONPATH="src"; python -m pipeline.predict
$env:PYTHONPATH="src"; python -m pipeline.generate_cf
$env:PYTHONPATH="src"; python -m pipeline.cf_metrics
$env:PYTHONPATH="src"; python -m pipeline.case_builder --pretty
$env:PYTHONPATH="src"; python scripts/run_metrics_only.py
$env:PYTHONPATH="src"; python scripts/visualize_metrics_only.py
$env:PYTHONPATH="src"; python scripts/run_debate.py --single-llm
```
