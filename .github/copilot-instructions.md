# Copilot Instructions

## Project Overview

Multi-agent adversarial evaluation of counterfactual explanations for the **Adult Income** binary classification task (OpenML, 48 842 rows, 14 features, target: ≤50K / >50K). Three evaluation strategies are compared: **metrics-only**, **single-LLM**, and **multi-agent debate** (Prosecutor → Defense → Expert Witness → Judge).

## Architecture — Three Layers

```
ML Pipeline (src/*.py)          →  Bridge (src/case_builder.py)  →  Evaluation (src/agents/ + src/run_debate.py)
train → predict → generate_cf      CSV/JSON → case dicts             AutoGen SelectorGroupChat debate
→ cf_metrics                                                         or single-LLM baseline
```

- **ML pipeline** scripts are standalone (no classes), each reads upstream artifacts from `results/` or `models/`.
- **Bridge layer** (`case_builder.py`) converts pipeline CSVs into a `cases.json` array consumed by the debate system. Each case has multiple CFs per individual (not one).
- **Evaluation layer** (`src/agents/`) is an AutoGen package with clean separation: `config.py` (LLM providers), `agents.py` (5 agent definitions), `debate.py` (orchestration), `prompts.py` (issue taxonomy), `utils.py` (JSON parsing, cost, transcripts).

## Running Scripts

**All scripts must run from the repo root with `PYTHONPATH=src`:**

```powershell
$env:PYTHONPATH="src"; python src/train.py
$env:PYTHONPATH="src"; python src/predict.py
$env:PYTHONPATH="src"; python src/generate_cf.py
$env:PYTHONPATH="src"; python src/cf_metrics.py
$env:PYTHONPATH="src"; python src/case_builder.py --pretty
$env:PYTHONPATH="src"; python src/run_debate.py --verbose
```

Pipeline order matters — each script depends on outputs from previous steps. The full chain is: `explore_data → train → predict → generate_cf → cf_metrics → case_builder → run_debate`. (`explore_data` is optional but recommended before taxonomy work.)

## Key Data Flow

| Artifact | Produced by | Consumed by |
|---|---|---|
| `results/feature_catalog.json` | `explore_data.py` | Taxonomy design, agent prompt calibration (reference) |
| `models/logistic_regression.joblib` | `train.py` | `predict.py`, `generate_cf.py` |
| `results/unfavorable_samples.csv` | `predict.py` | `generate_cf.py`, `case_builder.py` |
| `results/counterfactuals.csv` | `generate_cf.py` | `cf_metrics.py`, `case_builder.py` |
| `results/generation_policy.json` | `generate_cf.py` | `case_builder.py`, documentation/debugging |
| `results/cf_metrics_per_instance.csv` | `cf_metrics.py` | `case_builder.py` |
| `results/cases.json` | `case_builder.py` | `run_debate.py` |

## Critical Conventions

- **Model**: Logistic Regression (chosen over XGBoost for DiCE compatibility). Saved as a full sklearn `Pipeline` (preprocessor + classifier) via joblib.
- **Feature policy**: Centralized in `src/feature_policy.py`. Raw `education` is excluded from model training because it duplicates `education-num`. DiCE may vary `age`, `education-num`, `workclass`, `occupation`, `hours-per-week`, `capital-gain`, and `capital-loss`; frozen features are `fnlwgt`, `marital-status`, `relationship`, `race`, `sex`, and `native-country`. Raw `education` is synchronized from `education-num` as a derived display label after generation.
- **DiCE method**: `"genetic"` with the reference/default genetic weights stored in `DICE_DEFAULT_GENETIC_KWARGS`. Per-instance `permitted_range` values are derived from empirical distributions and saved to `results/generation_policy.json`.
- **JSON serialisation**: Always convert numpy/pandas types to plain Python before JSON dump. Use the `_safe_python()` pattern from `case_builder.py` (handles `np.integer`, `np.floating`, `np.bool_`, `NaN → None`).
- **Verdict format**: Judge outputs JSON inside a ` ```json ` fenced block followed by `VERDICT_COMPLETE` on its own line. The parser in `agents/utils.py` handles fenced blocks, bare JSON, and inline objects as fallbacks.
- **Issue taxonomy**: Defined in `src/agents/prompts.py` as `ISSUE_TAXONOMY` dict (snake_case labels -> descriptions). Constraint violations are not scored labels; they are technical warnings for frozen-feature or permitted-range failures.

## Agent System (AutoGen)

- Uses `autogen-agentchat` `SelectorGroupChat` with either `round_robin` (deterministic) or `auto` (LLM-selected) speaker strategies.
- Termination: `TextMentionTermination("VERDICT_COMPLETE") | MaxMessageTermination(...)`.
- Expert Witness receives **pre-computed real DiCE metrics** — never ask the LLM to simulate or invent metric values.
- LLM providers configured via `.env` at repo root (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`). Provider resolution: CLI arg → env var `LLM_PROVIDER` → default `"groq"`.
- Groq models require explicit `model_info` dict because AutoGen doesn't recognise them natively.

## Placeholders & In-Progress Areas

- `ground_truth_issues` in cases is always `[]` — no labeling function yet. `case_builder.build_cases()` accepts a `label_fn` callback for this.
- SHAP integration is planned but not implemented.
- `ground_truth_issues` still has no labeling function; heuristic labels are computed deterministically but are not a human gold standard.
- Focused foundation tests live in `tests/test_foundation_policy.py`.
