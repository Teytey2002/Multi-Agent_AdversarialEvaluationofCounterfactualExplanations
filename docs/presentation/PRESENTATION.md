# Meeting Presentation — Session Summary
## Multi-Agent Adversarial Evaluation of Counterfactual Explanations

> **Format:** Stand-alone script. Read section by section. Each section maps to one commit or logical milestone.

---

## 🗂️ Where We Started

Before this session the repo already had Theo's initial version (`7d27687`) — a working ML pipeline with Logistic Regression, Random Forest, and XGBoost trained on the Adult Income dataset.

**The problem:** the pipeline produced models and counterfactuals, but nothing that *evaluated* those counterfactuals in a meaningful, explainable way. The multi-agent debate system existed only as a separate PoC (`autogen_poc/`), disconnected from the real data.

The goal for this session was to close that gap end-to-end.

---

## 🧹 Step 1 — Cleanup & Consolidation (`1ba0cba`)

**What we did:**
- Removed Random Forest and XGBoost from the pipeline. The DiCE counterfactual library requires gradient-compatible models, so Logistic Regression is the only viable choice.
- Added `.gitignore` to keep generated artefacts (`models/`, `results/debate_outputs/`) out of version control.
- Regenerated fresh pipeline results using the cleaned codebase.

**Key decision — why only Logistic Regression?**

> DiCE's `genetic` and `gradient` optimisers require the model to return differentiable or at least queryable probability scores in a predictable way. The sklearn `Pipeline` (preprocessor + LR) satisfies this cleanly. XGBoost required extra wrappers and Random Forest produced unreliable gradient estimates.

---

## 🔧 Step 2 — Audit Fixes (`3727825`)

**What we fixed** in `src/pipeline/cf_metrics.py`, `src/pipeline/generate_cf.py`, `src/pipeline/predict.py`:

| Fix | Why |
|-----|-----|
| NaN filtering in `predict.py` | Rows with `?` in `workclass`/`occupation` crashed DiCE's genetic search |
| MAD fallback in `cf_metrics.py` | `capital-gain` has median = 0 → MAD = 0 → division by zero. Fallback: `std`, then `1.0` |
| `cf_confidence` field added | Each counterfactual now carries the model's confidence in the flipped class |
| `fnlwgt` declared continuous in DiCE | Even though it is immutable, DiCE's genetic solver explodes if it treats a feature with 28K unique values as categorical |

These fixes were necessary before we could trust the metrics passed to the evaluation agents.

---

## 🌉 Step 3 — Bridge Layer: `case_builder.py` (`8343d15`)

**What it does:** converts pipeline CSV outputs into a structured JSON format that the AutoGen debate system can consume.

**Input files:**
- `results/unfavorable_samples.csv` — the individuals to evaluate
- `results/counterfactuals.csv` — DiCE-generated CFs
- `results/cf_metrics_per_instance.csv` — pre-computed DiCE quality metrics

**Output:** `results/cases.json` — an array of 10 case objects.

**One case looks like this (simplified):**
```json
{
  "case_id": 0,
  "domain": "Adult Income",
  "original": { "age": 42, "sex": "Male", "occupation": "Craft-repair", ... },
  "prediction": "<=50K",
  "prediction_confidence": 0.71,
  "true_label": "<=50K",
  "is_false_negative": false,
  "counterfactuals": [
    {
      "cf_index": 0,
      "cf_confidence": 0.63,
      "features_changed": ["occupation", "hours-per-week"],
      "changes_summary": {
        "occupation": { "from": "Craft-repair", "to": "Exec-managerial" },
        "hours-per-week": { "from": 40, "to": 50 }
      }
    }
  ],
  "metrics": {
    "validity": 1.0,
    "sparsity": 0.857,
    "continuous_proximity": -0.42,
    "categorical_proximity": 0.91
  },
  "ground_truth_issues": []
}
```

**Key design decision:** each case holds **multiple CFs** per individual (DiCE generates 5 by default), not one. This is important — the debate agents evaluate the *set* of counterfactuals, which allows them to reason about diversity and consistency across suggestions.

> `ground_truth_issues` is currently always `[]` — that is Ivan's placeholder. The `build_cases()` function accepts a `label_fn` callback so Ivan can inject his taxonomy labels when ready.

---

## 🤖 Step 4 — AutoGen Debate System (`225142b`)

This is the main integration. We adapted the standalone PoC into a production-ready `src/agents/` package.

### Package structure

```
src/agents/
├── __init__.py       ← public API re-exports
├── config.py         ← LLM provider resolution (Groq / Gemini / OpenAI)
├── prompts.py        ← issue taxonomy (PLACEHOLDER — Ivan replaces this)
├── agents.py         ← 5 agent definitions
├── debate.py         ← SelectorGroupChat orchestration
└── utils.py          ← verdict parsing, transcripts, cost, agreement
```

### The 5 agents

| Agent | Role |
|-------|------|
| **Prosecutor** | Attacks CFs — flags immutable feature changes, low confidence, unrealistic jumps |
| **Defense** | Defends useful CFs — highlights sparsity, actionability, diversity |
| **Expert_Witness** | Technical analysis of real DiCE metrics and deterministic heuristic evidence |
| **Judge** | Synthesises the debate → structured JSON verdict |
| **Single_Evaluator** | Same task as Judge but solo — our **baseline** for comparison |

### Key design decisions vs PoC

1. **Multi-CF schema** — `_build_case_prompt()` in `debate.py` formats the `counterfactuals[]` array with per-CF confidence, changed features, and changes summary. The PoC used mock single-CF data.

2. **Expert Witness evidence** — the Expert Witness analyses *real* DiCE metrics from `cf_metrics.py` plus deterministic heuristic evidence from `heuristics.py`. No quantitative evidence is invented by the LLM.

3. **Placeholder taxonomy** — `prompts.py` defines `ISSUE_TAXONOMY` with 7 default labels (`immutable_feature_change`, `proxy_feature_change`, `unrealistic_change`, etc.). All agent system prompts inject this shared vocabulary. Ivan replaces this file with his research-grounded taxonomy.

4. **Verdict format** — the Judge outputs a fenced ` ```json ` block followed by `VERDICT_COMPLETE`. The parser in `utils.py` handles fenced blocks, bare JSON, and inline fallbacks.

### LLM providers

Configured via `.env` at the repo root. Provider resolution order: CLI arg → `LLM_PROVIDER` env var → default (`groq`).

| Provider | Default model | Cost tier |
|----------|--------------|-----------|
| Groq | `llama-3.1-8b-instant` | Free tier |
| Gemini | `gemini-2.5-flash` | Very cheap |
| OpenAI | `gpt-4.1-mini` | Paid |

> **Note for Groq:** AutoGen doesn't recognise Groq models natively, so `config.py` injects an explicit `model_info` dict to declare capability flags.

### CLI entry point: `scripts/run_debate.py`

```powershell
# Multi-agent debate, all 10 cases, Groq default
$env:PYTHONPATH="src"; python scripts/run_debate.py

# Single-LLM baseline, case 0 only, verbose output
$env:PYTHONPATH="src"; python scripts/run_debate.py --single-llm --case-ids 0 --verbose

# Gemini, auto speaker selection
$env:PYTHONPATH="src"; python scripts/run_debate.py --speaker-selection auto
```

**Outputs saved to** `results/debate_outputs/<model>/<mode>_<timestamp>/`:
- `<mode>_results.json` — all verdicts, agreement metrics, costs
- `transcripts/case_XX_transcript.md` — per-case Markdown transcript

---

## 🔍 Step 5 — Feature Catalog: `explore_data.py` (`2598308`)

**What it does:** catalogues every feature's real-world value distribution from the full 48,842-row dataset.

**Output:** `results/feature_catalog.json`

Example entries:
```json
"capital-gain": {
  "type": "numerical",
  "min": 0.0, "max": 99999.0, "mean": 1079.07, "median": 0.0, "std": 7452.02
},
"occupation": {
  "type": "categorical",
  "n_unique": 15,
  "values": { "Exec-managerial": 6086, "Craft-repair": 6112, ... }
}
```

**Why this matters:**
- **Taxonomy grounding** — knowing that `capital-gain` median is 0 (over 50% of people have zero capital gains) makes it concrete why a CF that suggests "gain $99,999" is unrealistic.
- **Agent prompt calibration** — agents can reference actual value counts when assessing feasibility.
- **Box constraint validation** — compare DiCE's `permitted_range` (e.g. `hours-per-week: [20, 50]`) against the real range `[1, 99]`.

> `fnlwgt` is explicitly skipped — it has ~28K unique values and is a census sampling weight with no taxonomic meaning.

---

## 📖 Step 6 — Documentation (`fdbeb37`, `8d793f8`, `4b6697f`)

Documentation commits cleaned up and extended `docs/architecture/scripts.md` and `README.md`:

- **`4b6697f`** — Updated `README.md` to remove all Random Forest / XGBoost references; added rationale for choosing Logistic Regression over tree-based models for DiCE compatibility.

- **`fdbeb37`** — Added `explore_data.py` to the script reference as section `1b`, inserted it into the official run order, and documented why the feature catalog matters for downstream evaluation.

- **`8d793f8`** — Extended `cf_metrics.py` documentation with a **beginner-friendly explanation of MAD normalisation**, including a worked numerical example showing why a "10-year age change" and a "$2,000 capital-gain change" both normalise to the same distance of 2.0 MADs.

---

## 📐 Architecture Summary

```
load_adult_dataset()
      │
      ├── explore_data.py  ──────────────────►  results/feature_catalog.json
      │
      ▼
   train.py  ──────────────►  models/logistic_regression.joblib
      │
   predict.py  ────────────►  results/unfavorable_samples.csv
      │
   generate_cf.py  ─────────►  results/counterfactuals.csv
      │
   cf_metrics.py  ──────────►  results/cf_metrics_per_instance.csv
      │
   case_builder.py  ────────►  results/cases.json
      │
   run_debate.py
      │
      ├── [Multi-agent]   Prosecutor → Defense → Expert_Witness → Judge
      └── [Single-LLM]    Single_Evaluator
                                │
                    results/debate_outputs/
                    ├── <model>_multi_agent_latest.json
                    └── <model>/multi_agent_<ts>/transcripts/
```

---

## ✅ What Is Ready

| Component | Status |
|-----------|--------|
| ML pipeline (train → predict → generate_cf → cf_metrics) | ✅ Complete |
| `case_builder.py` bridge | ✅ Complete |
| `explore_data.py` + `feature_catalog.json` | ✅ Complete |
| `src/agents/` package (all 5 modules) | ✅ Complete |
| `scripts/run_debate.py` CLI | ✅ Complete |
| LLM provider support (Groq / Gemini / OpenAI) | ✅ Complete |
| Issue taxonomy | ⏳ Placeholder — Ivan delivers finalised version |
| `ground_truth_issues` labelling | ⏳ Pending — `label_fn` callback in `case_builder.py` |
| Calculated metric/heuristic evidence | ✅ Passed through `cases.json` |
| Smoke test / full run | 🔜 Ready to run — API keys in `.env` |

---

## 🔜 What Comes Next

1. **Ivan** replaces `src/agents/prompts.py` with the finalised taxonomy.
2. **Smoke test** — run `python scripts/run_debate.py --case-ids 0 --verbose` and verify a complete debate transcript + JSON verdict.
3. **Full run** — all 10 cases, both modes (`multi_agent` and `single_llm`), compare verdicts.
4. **Ground truth labelling** — use the `label_fn` hook in `case_builder.py` to assign expected issue labels to cases.
5. **Agreement metrics** — `compute_agreement()` in `agents/utils.py` is already implemented; plug in ground truth to get detection rate and false-positive rate.

---

*Document generated from commits `1ba0cba` → `3727825` → `8343d15` → `225142b` → `4b6697f` → `2598308` → `fdbeb37` → `8d793f8`.*
