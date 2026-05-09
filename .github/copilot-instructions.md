# Copilot Instructions

## Project Overview

Multi-agent adversarial evaluation of counterfactual explanations for the **Adult Income** binary classification task (OpenML, 48 842 rows, 14 features, target: ≤50K / >50K). Three evaluation strategies are compared: **metrics-only**, **single-LLM**, and **multi-agent debate** (Prosecutor → Defense → Expert Witness → Judge).

## Architecture — Three Layers

```
ML Pipeline (src/pipeline/)      →  Bridge (pipeline.case_builder)  →  Evaluation
train → predict → generate_cf       CSV/JSON → case dicts              metrics-only baseline
→ cf_metrics                                                          or AutoGen single/multi-agent
```

- **ML pipeline** modules live in `src/pipeline/`; each reads upstream artifacts from `results/` or `models/`.
- **Bridge layer** (`src/pipeline/case_builder.py`) converts pipeline CSVs into a `cases.json` array consumed by the debate system. Each case has multiple CFs per individual (not one).
- **Evaluation layer** includes `src/evaluators/metrics_only.py` for the deterministic non-LLM baseline and `src/agents/` for AutoGen single-LLM / multi-agent runs. The AutoGen package has clean separation: `config.py` (Groq-only LLM configuration), `agents.py` (5 agent definitions), `debate.py` (orchestration), `prompts.py` (issue taxonomy), `utils.py` (JSON parsing, cost, transcripts).

## Running Scripts

**All scripts must run from the repo root with `PYTHONPATH=src`:**

```powershell
$env:PYTHONPATH="src"; python -m pipeline.explore_data
$env:PYTHONPATH="src"; python -m pipeline.train
$env:PYTHONPATH="src"; python -m pipeline.predict
$env:PYTHONPATH="src"; python -m pipeline.generate_cf
$env:PYTHONPATH="src"; python -m pipeline.cf_metrics
$env:PYTHONPATH="src"; python -m pipeline.case_builder --pretty
$env:PYTHONPATH="src"; python scripts/run_metrics_only.py
$env:PYTHONPATH="src"; python scripts/run_debate.py --verbose
```

Pipeline order matters - each script depends on outputs from previous steps. The full chain is: `explore_data → train → predict → generate_cf → cf_metrics → case_builder → run_metrics_only / run_debate`. (`explore_data` is optional but recommended before taxonomy work.)

## Key Data Flow

| Artifact | Produced by | Consumed by |
|---|---|---|
| `results/feature_catalog.json` | `explore_data.py` | Taxonomy design, agent prompt calibration (reference) |
| `models/logistic_regression.joblib` | `train.py` | `predict.py`, `generate_cf.py` |
| `results/unfavorable_samples.csv` | `predict.py` | `generate_cf.py`, `case_builder.py` |
| `results/counterfactuals.csv` | `generate_cf.py` | `cf_metrics.py`, `case_builder.py` |
| `results/generation_policy.json` | `generate_cf.py` | `case_builder.py`, documentation/debugging |
| `results/cf_metrics_per_instance.csv` | `cf_metrics.py` | `case_builder.py` |
| `results/cases.json` | `case_builder.py` | `run_metrics_only.py`, `run_debate.py` |

## Critical Conventions

- **Model**: Logistic Regression (chosen over XGBoost for DiCE compatibility). Saved as a full sklearn `Pipeline` (preprocessor + classifier) via joblib.
- **Feature policy**: Centralized in `src/policy/feature_policy.py`. Raw `education` is excluded from model training because it duplicates `education-num`. DiCE may vary `age`, `education-num`, `workclass`, `occupation`, `hours-per-week`, `capital-gain`, and `capital-loss`; frozen features are `fnlwgt`, `marital-status`, `relationship`, `race`, `sex`, and `native-country`. Raw `education` is synchronized from `education-num` as a derived display label after generation.
- **DiCE method**: `"genetic"` with the reference/default genetic weights stored in `DICE_DEFAULT_GENETIC_KWARGS`. Per-instance `permitted_range` values are derived from empirical distributions and saved to `results/generation_policy.json`.
- **JSON serialisation**: Always convert numpy/pandas types to plain Python before JSON dump. Use the `_safe_python()` pattern from `case_builder.py` (handles `np.integer`, `np.floating`, `np.bool_`, `NaN → None`).
- **Verdict format**: Judge outputs JSON inside a ` ```json ` fenced block followed by `VERDICT_COMPLETE` on its own line. The parser in `agents/utils.py` handles fenced blocks, bare JSON, and inline objects as fallbacks.
- **Issue taxonomy**: Defined in `src/agents/prompts.py` as `ISSUE_TAXONOMY` dict (snake_case labels -> descriptions). Constraint violations are not scored labels; they are technical warnings for frozen-feature or permitted-range failures.

## Agent System (AutoGen)

- Uses `autogen-agentchat` `SelectorGroupChat` with either `round_robin` (deterministic) or `auto` (LLM-selected) speaker strategies.
- Termination: `TextMentionTermination("VERDICT_COMPLETE") | MaxMessageTermination(...)`.
- Expert Witness receives **pre-computed real DiCE metrics** — never ask the LLM to simulate or invent metric values.
- LLM execution is Groq-only. Configure `.env` with `GROQ_API_KEY`; optional overrides are `GROQ_MODEL` and `GROQ_BASE_URL`.
- Default model: `llama-3.1-8b-instant`. Official Groq Free Plan limits for this model are 30 RPM, 14.4K RPD, 6K TPM, and 500K TPD.
- Groq is called through AutoGen's OpenAI-compatible client, so the dependency name includes `openai` even though no OpenAI provider is configured.

## Placeholders & In-Progress Areas

- Draft reference labels live in `annotations/ground_truth_labels.json` and are injected into `results/cases.json` by `case_builder.py`.
- The current ground-truth labels are an initial human-perspective draft for review; they are not an external gold standard.
- Focused foundation tests live in `tests/test_foundation_policy.py`.
