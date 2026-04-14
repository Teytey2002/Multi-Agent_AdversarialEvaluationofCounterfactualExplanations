# Multi-Agent Framework Choice: AutoGen vs CrewAI

> **Date:** April 10, 2026
> **Authors:** Daniel (AutoGen PoC), Ivan (CrewAI PoC)
> **Context:** "Multi-Agent Adversarial Evaluation of Counterfactual Explanations"
> **Purpose:** Choose one framework for the final integrated pipeline

---

## 1 · Decision Summary

| | **AutoGen** | **CrewAI** |
|---|---|---|
| **Recommendation** | ✅ **Adopt** | ❌ Decline (borrow one design idea) |

**Verdict:** Adopt **AutoGen** as the production framework for the final pipeline. Port one key design pattern from CrewAI — the **pre-computed metrics injection** approach — into the AutoGen codebase.

The rest of this document presents the full evidence behind this recommendation.

---

## 2 · What Was Built

### 2.1 AutoGen PoC (`autogen_poc/`)

A complete experimental harness with:

- 6 Python files (~1,000 lines total)
- 4 specialised agents (Prosecutor, Defense, Expert Witness, Judge)
- 10 mocked cases across 3 domains (loan, hiring, credit scoring)
- Two evaluation modes: **multi-agent debate** and **single-LLM baseline**
- Full scoring pipeline: detection rate, false-positive rate, exact match
- Results for **4 models** (llama-3.1-8b, llama-4-scout-17b, qwen3-32b, llama-3.3-70b)
- Timestamped outputs, Markdown transcripts, per-model and cross-model charts
- CLI with 9 configurable flags
- Multi-provider support (Groq, Gemini, OpenAI)

### 2.2 CrewAI PoC (`CrewAI_alt/`)

A concise prototype with:

- 2 Python files (~160 lines total)
- 4 agents (same role design: Expert, Prosecutor, Defense, Judge)
- 3 mocked cases (loan, hiring, university admission)
- One evaluation mode: sequential multi-agent pipeline
- No scoring against ground truth
- Python heuristics pre-computed before agents run
- Pydantic-enforced JSON output schema
- Groq provider only

---

## 3 · Head-to-Head Comparison

### 3.1 Architecture & Modularity

| Dimension | AutoGen PoC | CrewAI PoC |
|---|---|---|
| File count | 6 (config, agents, debate, utils, mock_data, run_experiment) | 2 (main.py, heuristics.py) |
| Separation of concerns | Strong — each file has one job | Weak — agents, tasks, and orchestration all in `main.py` |
| Async support | ✅ Native async with sync wrappers | ❌ Synchronous only |
| Extensibility | High — swap mock_data.py for a real loader | Low — hardcoded `cases.json` load inline |

**Edge: AutoGen.** The modular structure means we can replace one component (e.g. mock data → real DiCE data) without touching the rest of the pipeline.

### 3.2 Conversation Control

| Dimension | AutoGen | CrewAI |
|---|---|---|
| Turn-taking | `SelectorGroupChat` with pluggable `selector_func` | `Process.sequential` (fixed task order) |
| Options | Round-robin, LLM-selected (`auto`), custom functions | Sequential or hierarchical (built-in) |
| Termination | Composable conditions: `TextMention \| MaxMessages` | Implicit — crew stops after last task |
| Multi-round debate | ✅ Configurable `--max-rounds` (agents speak multiple times) | ❌ Each agent speaks exactly once |

**Edge: AutoGen.** The ability to run multi-round debates where agents respond to each other's arguments is central to the adversarial evaluation thesis. CrewAI's sequential one-shot-per-agent model cannot produce a real back-and-forth debate.

### 3.3 Output Enforcement

| Dimension | AutoGen | CrewAI |
|---|---|---|
| JSON enforcement | Custom robust parser in `utils.py` (handles fenced blocks, bare JSON, edge cases) | `output_json=JudgeVerdict` Pydantic model — framework-level enforcement |
| Reliability | High (fallback strategies) but ultimately depends on LLM compliance | High — CrewAI retries internally until valid JSON is returned |

**Edge: CrewAI** (minor). Pydantic `output_json` is elegant. However, the AutoGen PoC's parser already handles every observed failure mode across 4 models × 10 cases × 2 modes = 80 verdict extractions, so the practical gap is small.

### 3.4 Provider Support

| Dimension | AutoGen | CrewAI |
|---|---|---|
| Mechanism | `OpenAIChatCompletionClient` with custom `base_url` per provider | `litellm` under the hood — built-in multi-provider routing |
| Setup code | ~80 lines in `config.py` | One string: `llm='groq/llama-3.1-8b-instant'` |
| Providers tested | Groq, Gemini, OpenAI | Groq only |
| Model metadata | Manual `model_info` dicts | Automatic via litellm |

**Edge: CrewAI** (minor). litellm's provider abstraction is cleaner. But AutoGen's approach already works for 3 providers and the extra config code is a one-time cost.

### 3.5 Experimental Infrastructure

This is the **decisive dimension** for a research project.

| Capability | AutoGen PoC | CrewAI PoC |
|---|---|---|
| Ground truth & scoring | ✅ Detection rate, FP rate, exact match per case | ❌ None |
| Single-LLM baseline | ✅ Built-in `--single-llm` mode | ❌ Not implemented |
| Multi-model comparison | ✅ Cross-model charts, per-case heatmaps | ❌ Single model only |
| Transcript saving | ✅ Per-case Markdown files | ❌ Console output only |
| Cost estimation | ✅ Per-case token counts and USD estimates | ❌ None |
| Timestamped runs | ✅ `outputs/<model>/<mode>_<timestamp>/` | ❌ None |
| CLI configurability | ✅ 9 flags (provider, model, rounds, delay, etc.) | ❌ Hardcoded |
| Reproducibility | ✅ Config saved in every results JSON | ❌ No config recorded |
| Rate-limit handling | ✅ `--delay`, `max_retries`, per-provider defaults | ⚠️ `max_rpm=1` (very conservative) |

**Edge: AutoGen** (dominant). The project's scientific goal is to **compare three evaluation strategies** (metrics-only, single-LLM, multi-agent). AutoGen already implements two of them with full scoring. CrewAI would require building this entire infrastructure from scratch.

### 3.6 Results & Evidence

| Dimension | AutoGen PoC | CrewAI PoC |
|---|---|---|
| Models tested | 4 (8B, 17B, 32B, 70B) | 1 (8B) |
| Cases evaluated | 10 × 4 models × 2 modes = ~80 evaluations | 3 cases × 1 run |
| Documented findings | Detailed `RESULTS.md` with tables, per-case breakdowns, rate-limit analysis | `counterfactual_pipeline_summary.md` — case analysis but no experimental metrics |
| Key insight discovered | Single-LLM outperforms multi-agent on mock data; debate amplifies false positives | None (no scoring to measure) |

**Edge: AutoGen.** The PoC already generated publishable findings.

---

## 4 · The Integration Challenge

Neither PoC currently connects to the real ML pipeline. Here is what integration requires, and how each framework handles it.

### 4.1 What the Main Pipeline Produces

| Artifact | File | Format |
|---|---|---|
| Original unfavorable instances | `results/unfavorable_samples.csv` | 14 features + prediction + proba + true_label |
| Generated counterfactuals | `results/counterfactuals.csv` | 14 features + income (prediction) + original_index |
| Counterfactual metrics | `results/cf_metrics.csv` | original_index, sparsity, proximity |
| Trained model | `models/logistic_regression.joblib` (should become the main model) | sklearn Pipeline |
| Model evaluation | `results/*_metrics.json` | accuracy, precision, recall, F1 |

### 4.2 Integration Steps Required

| Step | Description | AutoGen readiness | CrewAI readiness |
|---|---|---|---|
| 1. Load real data | Read CSVs, pair each CF with its original | Replace `mock_data.py` → straightforward | Rewrite inline loading in `main.py` |
| 2. Compute real metrics | Sparsity, proximity, SHAP, plausibility | Add to data loader; inject into case prompt | Already has `heuristics.py` pattern ✅ |
| 3. Format as cases | Convert (original, CF, metrics) → case dict | Case schema already exists in `mock_data.py` | Case schema is in `cases.json` |
| 4. Run evaluation | Multi-agent + single-LLM + metrics-only | 2 of 3 modes already built | 1 mode only |
| 5. Score results | Compare verdicts | `compute_agreement()` exists; needs adaptation for real data (no planted ground truth → human labels) | Must be built from scratch |
| 6. Visualise | Charts, transcripts, reports | Already built | Must be built from scratch |

**Edge: AutoGen.** The only missing piece is the data loader. With CrewAI, nearly everything after the agent definitions must be written.

### 4.3 The One Thing to Borrow from CrewAI

CrewAI's `heuristics.py` demonstrates a powerful design pattern:

```
Pre-compute deterministic metrics in Python → Inject as structured text into agent prompts
```

In the AutoGen PoC, the Expert Witness currently **simulates** SHAP values and feasibility scores via the LLM, because the data is mocked. In the final pipeline, the Expert Witness should **receive real computed metrics** — sparsity, proximity, SHAP feature importances — as pre-computed evidence, just like CrewAI does.

This is not a framework feature; it's a **design decision** that can be adopted in AutoGen by enriching the case prompt with computed metrics before the debate starts.

---

## 5 · Risk Analysis

### 5.1 Risks of Choosing AutoGen

| Risk | Severity | Mitigation |
|---|---|---|
| API-level breaking changes (AutoGen is pre-1.0, currently 0.7.x) | Medium | Pin version in `requirements.txt`; the PoC already works on 0.4.9+ |
| More complex codebase to maintain | Low | Already modular; well-documented |
| No native Pydantic output enforcement | Low | Custom parser handles all observed cases; can add `instructor` library if needed |

### 5.2 Risks of Choosing CrewAI

| Risk | Severity | Mitigation |
|---|---|---|
| Must rebuild entire experimental infrastructure | **High** | Weeks of work duplicating what AutoGen PoC already has |
| No multi-round debate support | **High** | Would need to hack around `Process.sequential` or use `Process.hierarchical` (less control) |
| No single-LLM baseline mode | **High** | Must implement from scratch |
| Less documented for this specific project | Medium | `counterfactual_pipeline_summary.md` is thorough but covers analysis, not infrastructure |

---

## 6 · What the Final Integrated Pipeline Looks Like

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN ML PIPELINE (existing)                  │
│                                                                 │
│  data_loader → preprocessing → train → predict → generate_cf   │
│                                                                 │
│  Outputs:                                                       │
│    • unfavorable_samples.csv                                    │
│    • counterfactuals.csv                                        │
│    • models/*.joblib                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              BRIDGE LAYER (new — to be built)                   │
│                                                                 │
│  1. Load (original, CF) pairs from CSVs                         │
│  2. Compute real metrics: sparsity, proximity, SHAP             │
│  3. Identify feature changes and flag protected attributes      │
│  4. Format each pair as a structured "case" dict                │
│  5. Optionally attach human annotations for ground truth        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         EVALUATION LAYER (AutoGen — adapted from PoC)           │
│                                                                 │
│  Strategy 1: Metrics-only baseline                              │
│    → Verdict derived purely from computed sparsity/proximity/   │
│      SHAP thresholds (no LLM involved)                          │
│                                                                 │
│  Strategy 2: Single-LLM evaluator                               │
│    → One agent sees the case + metrics and produces a verdict   │
│                                                                 │
│  Strategy 3: Multi-agent adversarial debate                     │
│    → Prosecutor, Defense, Expert (with real metrics), Judge     │
│    → Multiple rounds of argumentation                           │
│    → Structured JSON verdict                                    │
│                                                                 │
│  Scoring & comparison across all three strategies               │
│  Transcripts, charts, reports                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7 · Concrete Next Steps

### Phase 1 — Fix the ML backbone (Theo)
- [ ] Switch main model from XGBoost to **Logistic Regression** for DiCE compatibility (gradient method)
- [ ] Re-generate counterfactuals with `method="gradient"` instead of `method="random"`
- [ ] Fix `cf_metrics.py` so each CF is compared against its correct original (not the first row of each group)
- [ ] Add SHAP integration for feature-level explanations

### Phase 2 — Build the bridge layer (all)
- [ ] Write a `case_builder.py` that loads real CSVs and produces case dicts matching the AutoGen schema
- [ ] Compute real metrics (sparsity, proximity, SHAP) per case and inject them into the case prompt
- [ ] Define a real issue taxonomy grounded in the Adult Income domain (protected attributes: `race`, `sex`, `native-country`; immutable: `age`; actionability constraints per feature)
- [ ] Decide on evaluation strategy: human annotations, synthetic ground truth, or inter-rater agreement

### Phase 3 — Adapt the AutoGen PoC (Daniel)
- [ ] Move `autogen_poc/` code into the main repo as `src/evaluation/`
- [ ] Replace `mock_data.py` with `case_builder.py`
- [ ] Update Expert Witness agent to present **real** computed metrics instead of simulated ones
- [ ] Add a metrics-only baseline mode (no LLM — verdict from thresholds alone)
- [ ] Update the issue taxonomy for the Adult Income domain
- [ ] Run experiments with real counterfactuals

### Phase 4 — Scale experiments
- [ ] Run single-LLM and multi-agent on real data across multiple models
- [ ] Compare all three strategies: metrics-only vs single-LLM vs multi-agent
- [ ] Collect transcripts and generate comparison visuals
- [ ] Write up findings for the thesis

---

## 8 · Summary Table

| Criterion | Winner | Margin |
|---|---|---|
| Modularity & code quality | AutoGen | Large |
| Conversation control (multi-round debate) | AutoGen | **Critical** |
| Output JSON enforcement | CrewAI | Small |
| Provider support simplicity | CrewAI | Small |
| Experimental infrastructure | AutoGen | **Dominant** |
| Single-LLM baseline | AutoGen | **Critical** (CrewAI has none) |
| Documentation & results | AutoGen | Large |
| Ease of integration with main pipeline | AutoGen | Large |
| Pre-computed metrics injection pattern | CrewAI | Small (design idea, not framework feature) |

**Final call:** AutoGen is the right choice. The experimental infrastructure gap alone would take weeks to close in CrewAI, and multi-round debate — the core of the thesis — is a native AutoGen capability that CrewAI lacks.

Borrow CrewAI's pre-computed metrics injection pattern. Thank Ivan for validating that the 4-agent courtroom design works in a second framework — that's independent confirmation of the architecture.

---

*Document generated from analysis of all three workspace projects on April 10, 2026.*
