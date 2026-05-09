# Pipeline Scripts — Per-Script Documentation

> Quick reference for pipeline modules in `src/pipeline/` and runnable entry points in `scripts/`.
> Run order: **data_loader → explore_data → preprocessing → models → train → predict → generate_cf → cf_metrics → case_builder → run_metrics_only / run_debate**
> (utils is a shared helper module; evaluators/ powers the deterministic baseline; agents/ is used by run_debate.)

All scripts must be executed from the **repo root** with `PYTHONPATH=src`:

```bash
# PowerShell
$env:PYTHONPATH="src"; python -m pipeline.<module>
$env:PYTHONPATH="src"; python scripts/<entrypoint>.py

# Bash / Linux
PYTHONPATH=src python -m pipeline.<module>
PYTHONPATH=src python scripts/<entrypoint>.py
```

---

## 1. `data_loader.py`

**Purpose** — Load the Adult Income dataset from OpenML and convert the target to binary.

### Workflow
1. Calls `sklearn.datasets.fetch_openml(name="adult", version=2)` to download / cache the dataset.
2. Maps the target column: `<=50K → 0`, `>50K → 1`.
3. Returns `(X, y)` as a pandas DataFrame / Series.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `name` | `"adult"` | OpenML dataset identifier |
| `version` | `2` | Uses the 48 842-row version |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | Internet connection (first call) or local sklearn cache |
| **Output** | `X` — DataFrame (48 842 × 14 features), `y` — Series (binary 0/1) |

---

## 1b. `explore_data.py`

**Purpose** — Catalogue every feature's possible values to support taxonomy design and agent prompt grounding.

### Workflow
1. Loads the Adult dataset via `load_adult_dataset()`.
2. For each feature (except `fnlwgt` — a census sampling weight with ~28 K unique values):
   - **Categorical**: lists every unique value with its frequency count.
   - **Numerical**: computes min, max, mean, median, std, and unique-value count.
3. Also summarises the target distribution (`<=50K` / `>50K`).
4. Writes everything to a single JSON file.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `SKIP_FEATURES` | `{"fnlwgt"}` | Not meaningful for taxonomy design |

### CLI
```bash
$env:PYTHONPATH="src"; python -m pipeline.explore_data
```

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | OpenML Adult dataset (auto-downloaded / cached) |
| **Output** | `results/feature_catalog.json` — per-feature value catalog |

### Why this matters
The output provides a grounded reference for:
- **Issue taxonomy design** — knowing that `workclass` has 9 values (including `nan`) or that `capital-gain` has a median of 0 helps set realistic thresholds for what constitutes an "unrealistic change".
- **Agent prompt calibration** — agents can reference real value distributions instead of guessing.
- **Box constraint validation** — compare DiCE's `permitted_range` in `generate_cf.py` against actual data ranges (e.g. `hours-per-week` real range [1, 99] vs box constraint [20, 50]).

---

## 2. `preprocessing.py`

**Purpose** — Build a `ColumnTransformer` that handles numerical and categorical features.

### Workflow
1. Detects column types from the DataFrame (`object`/`category` → categorical, `int64`/`float64` → numerical).
2. Numerical pipeline: `SimpleImputer(strategy="median")` → `StandardScaler()`.
3. Categorical pipeline: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`.
4. Returns the unfitted `ColumnTransformer`.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Numerical imputation | `median` | Robust to outliers |
| Categorical imputation | `most_frequent` | Fills NaN categories with mode |
| OHE `handle_unknown` | `"ignore"` | Prevents errors on unseen categories at predict time |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `X` DataFrame (used only to detect column types) |
| **Output** | Unfitted `ColumnTransformer` (becomes part of the sklearn `Pipeline`) |

---

## 3. `models.py`

**Purpose** — Define the classifier used throughout the pipeline.

### Workflow
1. Returns a `(name, estimator)` tuple: `"logistic_regression"`, `LogisticRegression(max_iter=1000)`.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_iter` | `1000` | Ensures convergence on the high-dimensional OHE feature space |
| `random_state` | `42` (default) | Reproducibility |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | Optional `random_state` |
| **Output** | `("logistic_regression", LogisticRegression(...))` |

---

## 4. `train.py`

**Purpose** — Train the Logistic Regression pipeline and persist the model + evaluation metrics.

### Workflow
1. Loads dataset via `load_adult_dataset()`.
2. Splits 80 / 20 stratified train-test.
3. Builds the preprocessor, wraps it with the classifier in a sklearn `Pipeline`.
4. Fits the pipeline on the training set.
5. Evaluates on the test set (accuracy, precision, recall, F1, confusion matrix, classification report).
6. Saves the fitted pipeline as `.joblib` and metrics as `.json`.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `TEST_SIZE` | `0.2` | 80/20 split |
| `RANDOM_STATE` | `42` | Reproducibility for split + model |
| `MODELS_DIR` | `"models"` | Output directory for `.joblib` |
| `RESULTS_DIR` | `"results"` | Output directory for metrics JSON |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | OpenML Adult dataset (auto-downloaded) |
| **Output** | `models/logistic_regression.joblib`, `results/logistic_regression_metrics.json` |

### Latest metrics (pipeline run 2025-04-14)
| Metric | Value |
|--------|-------|
| Accuracy | 0.8524 |
| Precision | 0.7414 |
| Recall | 0.5885 |
| F1 | 0.6562 |

---

## 5. `predict.py`

**Purpose** — Load the trained model, predict on the full dataset, and sample unfavorable individuals.

### Workflow
1. Loads `models/logistic_regression.joblib`.
2. Predicts class + probability on the entire Adult dataset.
3. Filters rows with NaN values (`?` markers in workclass / occupation) to guarantee clean downstream input.
4. Filters *unfavorable* cases (predicted class = 0, i.e. ≤50 K).
5. Randomly samples `SAMPLE_SIZE` unfavorable individuals (`random_state=42`).
6. Adds an `is_false_negative` flag: `True` when the model predicted 0 but the true label was 1. **Why this matters:** Counterfactuals generated for individuals the model already misclassifies are inherently less trustworthy. The downstream evaluation agents use this flag to properly contextualize or discount explanations for individuals the model simply failed to understand.
7. Saves the sample to CSV.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_PATH` | `"models/logistic_regression.joblib"` | Trained pipeline |
| `SAMPLE_SIZE` | `10` | Number of unfavorable individuals to pick |
| `random_state` | `42` | Reproducible sampling |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `models/logistic_regression.joblib`, OpenML Adult dataset |
| **Output** | `results/unfavorable_samples.csv` (10 rows: features + prediction + proba + true_label + is_false_negative) |

---

## 6. `generate_cf.py`

**Purpose** — Generate counterfactual explanations using DiCE's genetic algorithm.

### Workflow
1. Loads the trained model and the full dataset.
2. Cleans the dataset: replaces `"?"` with NaN, drops incomplete rows.
3. Loads `unfavorable_samples.csv`, applies the same cleaning (rows with NaN are dropped).
4. Builds DiCE objects: `dice_ml.Data`, `dice_ml.Model(backend="sklearn")`, `dice_ml.Dice(method="genetic")`.
5. For each clean instance, generates `TOTAL_CFS` counterfactuals with:
   - **Policy-defined actionable features** (`age`, `education-num`, `workclass`, `occupation`, `hours-per-week`, `capital-gain`, `capital-loss`).
   - **Per-instance empirical box constraints** from `feature_policy.build_permitted_range()`.
   - DiCE genetic default weights collected in `DICE_DEFAULT_GENETIC_KWARGS`.
   - Post-hoc sparsity via `posthoc_sparsity_param=0.1`.
6. Saves a structured CSV with `row_type` (original / counterfactual), `original_index`, `cf_rank`, and `cf_confidence` (model's P(class 1) for each row) columns.
7. Saves generation-policy metadata to `results/generation_policy.json`.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `TOTAL_CFS` | `4` | Number of CFs per instance |
| `DESIRED_CLASS` | `1` | Target: >50 K income |
| `method` | `"genetic"` | Best available for sklearn backend |
| `proximity_weight` | `0.2` | DiCE genetic default: keep CFs close to original |
| `sparsity_weight` | `0.2` | DiCE genetic default: prefer fewer feature changes |
| `diversity_weight` | `5.0` | DiCE genetic default: diversify the set of CFs |
| `categorical_penalty` | `0.1` | DiCE genetic default: penalty for categorical feature changes |
| `stopping_threshold` | `0.5` | Probability threshold to accept a CF |
| `posthoc_sparsity_param` | `0.1` | Linear sparsity post-processing |

#### Actionable features
| Feature | Rationale |
|---------|-----------|
| `age` | Long-term recourse horizon; may increase only |
| `education-num` | Long-term education recourse; may increase only and must be plausible with age |
| `workclass` | Job sector can change |
| `occupation` | Job role can change |
| `hours-per-week` | Working hours adjustable |
| `capital-gain` | Financial gains can vary |
| `capital-loss` | Financial losses can vary |

Frozen / excluded features (`race`, `sex`, `native-country`, `marital-status`, `relationship`, `fnlwgt`) are **not** allowed to vary. Raw `education` is excluded from model training and synchronized from `education-num` after generation as a derived display label.

#### Box constraints
Box constraints are no longer fixed constants. `feature_policy.build_permitted_range()` derives per-instance ranges from the data distribution and the individual's current value, then stores them in `results/generation_policy.json`.

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `models/logistic_regression.joblib`, `results/unfavorable_samples.csv`, OpenML Adult dataset |
| **Output** | `results/counterfactuals.csv` — structured table with originals + their CFs + `cf_confidence` per row |

---

## 7. `cf_metrics.py`

**Purpose** — Evaluate counterfactual quality using DiCE-paper metrics, particularly focusing on distances adjusted by MAD (Median Absolute Deviation).

### 💡 Beginner's Guide to "MAD Normalization"

**What is it?** MAD (Median Absolute Deviation) measures how spread out a feature's values are. It is similar to standard deviation, but it uses medians instead of averages, making it highly robust against extreme outliers (which are common in financial data like `capital-gain`).
**Why do we need it?** To measure if a counterfactual is "close" to the original person's profile, we have to calculate the distance between their feature values. But changing `age` by +10 years is vastly different from changing `capital-gain` by +$10. By dividing every raw change by that feature's MAD (i.e. "normalizing" it), we put all numerical features onto the exact same scale. A change of "1.0 MAD" represents a similar magnitude of shift regardless of the underlying unit.

**Mockup Numerical Example:**

Imagine we have 5 people with the following `age` values: `[30, 32, 35, 40, 50]`.
* **Step 1 (Find Median):** The median age is **35**.
* **Step 2 (Absolute Differences):** How far is each person's age from 35? `[5, 3, 0, 5, 15]`.
* **Step 3 (Median of Differences):** The median of those differences `[0, 3, 5, 5, 15]` is **5**.
So, the **MAD for Age is 5**.

Now, let's compare calculating distances:
* If a counterfactual changes a person's `age` from 35 to 45, the raw change is **10 years**.
  * *MAD-Normalized distance* = 10 changes ÷ MAD of 5 = **2.0 MADs**.
* Assume `capital-gain` has a **MAD of $1,000**. If a counterfactual changes it by **$2,000**.
  * *MAD-Normalized distance* = 2000 changes ÷ MAD of 1000 = **2.0 MADs**.

*Meaning:* Even though "10 years" and "$2,000" are completely different units, the normalization gives them both a distance of **2.0**. This proves to the evaluator agents that mathematically, both changes represent an equally "drastic" shift relative to how the data naturally varies in the real-world Adult Income Dataset!

### Workflow

1. Loads the full Adult dataset to compute feature types and MAD values for all numerical columns.
2. Reads `results/counterfactuals.csv`.
3. Groups rows by `original_index`, separates originals from counterfactuals.
4. For each instance, computes:
   - **Validity** — fraction of unique CFs that achieved the desired class.
   - **Continuous proximity** — average MAD-normalized distance (negative = closer is better; adjusting by MAD ensures `age` and `capital-gain` are compared fairly).
   - **Categorical proximity** — average fraction of unchanged categorical features.
   - **Sparsity** — `1 − (changed_features / total_features)`. Higher = sparser (fewer changes).
   - **Continuous diversity** — pairwise MAD-normalized distance between CFs.
   - **Categorical diversity** — pairwise fraction of differing categorical features.
   - **Count diversity** — pairwise fraction of differing features (any type).
5. Saves per-instance and global (averaged) metrics.

### Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| MAD normalization | Per-column MAD from full dataset | Sometimes MAD is exactly 0 (e.g. if >50% of people have $0 `capital-gain`). To avoid dividing by zero, the script safely falls back to standard deviation (`std`), then to `1.0` if both are 0. |
| Proximity sign | Negative (continuous) | Paper convention: closer to 0 is better |
| Sparsity convention | Higher is better | `1 − (changed / total)` |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `results/counterfactuals.csv`, OpenML Adult dataset |
| **Output** | `results/cf_metrics_per_instance.csv`, `results/cf_metrics_global.csv` |

### Metric interpretation (quick guide)
| Metric | Range | Better ↑/↓ |
|--------|-------|------------|
| Validity | [0, 1] | ↑ Higher |
| Continuous proximity | (−∞, 0] | ↑ Closer to 0 |
| Categorical proximity | [0, 1] | ↑ Higher |
| Sparsity | [0, 1] | ↑ Higher |
| Continuous diversity | [0, +∞) | ↑ Higher |
| Categorical diversity | [0, 1] | ↑ Higher |
| Count diversity | [0, 1] | ↑ Higher |

---

## 8. `utils.py`

**Purpose** — Shared utility functions for training and evaluation.

### Functions

| Function | Description |
|----------|-------------|
| `ensure_dir(path)` | Create directory if it doesn't exist (`os.makedirs`) |
| `evaluate_model(model, X_test, y_test)` | Compute accuracy, precision, recall, F1, confusion matrix, classification report |
| `save_model(model, filepath)` | Serialize a fitted sklearn pipeline with `joblib.dump` |
| `save_metrics(metrics, filepath)` | Write a metrics dict to JSON |

### Inputs / Outputs
Not run directly — imported by `train.py`.

---

## 9. `case_builder.py`

**Purpose** — Bridge layer that converts the ML pipeline's CSV outputs into structured JSON cases consumable by the AutoGen multi-agent debate system.

### Workflow
1. Reads pipeline artifacts:
   - `results/unfavorable_samples.csv` - the 10 sampled individuals.
   - `results/counterfactuals.csv` - originals + counterfactual rows (row_type / cf_rank).
   - `results/cf_metrics_per_instance.csv` - per-instance DiCE quality metrics.
   - `results/logistic_regression_metrics.json` - model evaluation metrics.
   - `annotations/ground_truth_labels.json` - draft human-perspective reference labels, when available.
2. For each sample, locates its CFs via `original_index`.
3. Computes `changes_summary` per CF: `{feature: {from: ..., to: ...}}` for every changed feature.
4. Assembles a case dict with:
   - Person profile (`original`), model prediction + confidence, true label, false-negative flag.
   - `counterfactuals` array (one entry per CF, each with features, features_changed, changes_summary, cf_confidence).
   - `metrics` block (validity, proximity, sparsity, diversity).
   - `model_info` (name, accuracy, precision, recall, F1).
   - `ground_truth_issues`, `ground_truth_by_cf`, and `ground_truth_source` from the annotation artifact.
5. Writes all cases to a single JSON file.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `--out` | `results/cases.json` | Output path (default) |
| `--pretty` | flag | Indent JSON for readability |

### CLI
```bash
$env:PYTHONPATH="src"; python -m pipeline.case_builder --pretty
```

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `results/unfavorable_samples.csv`, `results/counterfactuals.csv`, `results/cf_metrics_per_instance.csv`, `results/logistic_regression_metrics.json`, `annotations/ground_truth_labels.json` |
| **Output** | `results/cases.json` — array of case dicts (one per sampled individual) |

### Output schema (per case)
```json
{
  "case_id": 0,
  "domain": "income_prediction",
  "model_info": { "name": "...", "accuracy": 0.85, ... },
  "original": { "age": 22, "workclass": "Private", ... },
  "prediction": "<=50K",
  "prediction_confidence": 0.994,
  "true_label": "<=50K",
  "is_false_negative": false,
  "counterfactuals": [
    {
      "cf_rank": 0,
      "cf_confidence": 0.515,
      "features": { ... },
      "features_changed": ["workclass", "occupation", ...],
      "changes_summary": { "workclass": {"from": "Private", "to": "Federal-gov"}, ... }
    }
  ],
  "metrics": { "validity": 1.0, "continuous_proximity": -3.01, ... },
  "ground_truth_issues": ["fragile_counterfactual"],
  "ground_truth_by_cf": { "0": ["fragile_counterfactual"] },
  "ground_truth_source": { "file": "annotations/ground_truth_labels.json", ... }
}
```

### Differences from AutoGen PoC mock_data format
| PoC (mock_data.py) | case_builder.py | Notes |
|--------------------|-----------------|-------|
| `model_type` (string) | `model_info` (dict) | Includes real accuracy/precision/recall/F1 |
| `counterfactual` (single) | `counterfactuals` (array) | Multiple CFs per case with ranking |
| `features_changed` (top-level) | Per-CF | Each CF has its own changed features |
| — | `prediction_confidence` | Model P(class) for the original |
| — | `cf_confidence` | Model P(class 1) per CF |
| — | `true_label` / `is_false_negative` | Ground-truth info from dataset |
| — | `changes_summary` | Structured from/to diffs |
| — | `metrics` | Real DiCE quality metrics |

---

## 10. `evaluators/metrics_only.py` and `run_metrics_only.py`

**Purpose** - Deterministic non-LLM baseline that converts computed case evidence into the same verdict schema used by the LLM Judge.

### Workflow
1. Loads `results/cases.json`.
2. Reads case-level deterministic evidence:
   - `heuristic_summary.flagged_issues_union`
   - `heuristic_summary.constraint_violations_union`
   - `metrics` such as validity, proximity, sparsity, diversity
   - `is_false_negative`
3. Produces one verdict per case with:
   - `overall_assessment`
   - `flagged_issues`
   - `severity`
   - `confidence`
   - `reasoning_summary`
   - `recommended_action`
4. Saves timestamped and latest JSON outputs under `results/metrics_only_outputs/`.

### CLI
```bash
$env:PYTHONPATH="src"; python scripts/run_metrics_only.py

$env:PYTHONPATH="src"; python scripts/run_metrics_only.py --case-ids 0 3 5
```

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `results/cases.json` |
| **Output** | `results/metrics_only_outputs/metrics_only_<timestamp>/metrics_only_results.json` and `results/metrics_only_outputs/metrics_only_latest.json` |

### Important interpretation
This baseline is **not ground truth**. It is one competitor in the comparison. Human/team annotations in `ground_truth_issues` remain the reference labels when scoring metrics-only vs single-LLM vs multi-agent outputs.

---

## 11. `visualize_metrics_only.py` and `visualize_evaluations.py`

**Purpose** - Render dependency-free SVG dashboards from scored evaluation JSON files.

### Workflow
1. `visualize_metrics_only.py` keeps the original metrics-only-only dashboard entry point.
2. `visualize_evaluations.py` loads the latest metrics-only, single-LLM, and multi-agent outputs by default.
3. Computes visual comparison summaries:
   - issue recall, precision, F1, exact match
   - ground-truth vs system-predicted issue counts by label
   - per-case match / missed / extra / mixed status
   - cross-system comparison of precision, recall, F1, exact match, and issue divergence
4. Writes SVG figures to `docs/reports/figures/` by default so report visuals can be committed.

### CLI
```bash
$env:PYTHONPATH="src"; python scripts/visualize_metrics_only.py

$env:PYTHONPATH="src"; python scripts/visualize_metrics_only.py --input results/metrics_only_outputs/metrics_only_latest.json --output results/metrics_only_outputs/visuals/custom_summary.svg

$env:PYTHONPATH="src"; python scripts/visualize_evaluations.py

$env:PYTHONPATH="src"; python scripts/visualize_evaluations.py single --input results/debate_outputs/llama-3.1-8b-instant_single_llm_latest.json --system-name "Single LLM" --output docs/reports/figures/single_llm_summary.svg

$env:PYTHONPATH="src"; python scripts/visualize_evaluations.py compare --inputs results/metrics_only_outputs/metrics_only_latest.json results/debate_outputs/llama-3.1-8b-instant_single_llm_latest.json results/debate_outputs/llama-3.1-8b-instant_multi_agent_latest.json --system-names "Metrics-Only" "Single LLM" "Multi-Agent" --output docs/reports/figures/system_comparison_summary.svg
```

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | metrics-only, single-LLM, or multi-agent scored results JSON |
| **Output** | SVG visual summaries and comparison figures |

---

## 12. `agents/` package

**Purpose** - Multi-agent adversarial debate system for evaluating counterfactual explanations, adapted from the AutoGen PoC.

### Package structure

| Module | Description |
|--------|-------------|
| `__init__.py` | Package marker; re-exports `build_debate_agents`, `build_single_evaluator_agent`, `run_debate`, `run_single_llm` |
| `prompts.py` | **Issue taxonomy** (placeholder — Ivan replaces this) + `get_issue_guidance()` formatter |
| `agents.py` | 4 debate agents (Prosecutor, Defense, Expert\_Witness, Judge) + 1 single-LLM baseline (Single\_Evaluator) |
| `config.py` | `LLMConfig` dataclass, Groq-only model/API configuration, `build_model_client()` |
| `debate.py` | `run_debate()` / `run_single_llm()` — orchestrates `SelectorGroupChat` with round-robin or auto speaker selection |
| `utils.py` | `parse_judge_verdict()`, `serialise_message()`, `calculate_cost()`, `save_debate_transcript()`, `compute_agreement()` |

### Agent roles

| Agent | Role |
|-------|------|
| **Prosecutor** | Attacks CF quality — fairness, feasibility, actionability, low confidence |
| **Defense** | Defends useful CFs, narrows claims, highlights actionable features |
| **Expert\_Witness** | Technical analysis of real DiCE metrics, confidence scores, deterministic heuristic evidence, and feature-change feasibility |
| **Judge** | Synthesises debate → structured JSON verdict (`fair` / `unfair` / `ambiguous`) |
| **Single\_Evaluator** | Same task as Judge but without a debate (baseline comparison) |

### Key design decisions (vs PoC)
- **Multi-CF schema**: prompts handle `counterfactuals[]` array with per-CF `cf_confidence`, `features_changed`, `changes_summary`.
- **Expert Witness evidence**: analyses real DiCE metrics (validity, proximity, sparsity, diversity), confidence scores, and deterministic heuristic evidence instead of inventing quantitative evidence.
- **Issue taxonomy**: `prompts.py` contains the scored labels used by agents. Constraint violations are kept separate from scored issue labels.
- **Real data**: loads `results/cases.json` (from `case_builder.py`), not mock cases.

### Environment setup
Requires a `.env` file at the repo root with a Groq API key:

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

`GROQ_MODEL` and `GROQ_BASE_URL` are optional; the values above are the
defaults. The current experiment series is Groq-only. `llama-3.1-8b-instant`
is the default because the official Groq Free Plan limits are workable for
repeated case runs: 30 RPM, 14.4K RPD, 6K TPM, and 500K TPD. The code uses
AutoGen's OpenAI-compatible client because Groq exposes an OpenAI-compatible
endpoint; this is a client adapter, not an OpenAI provider configuration.

### Debate flow — how a single case is processed

This section walks through exactly what happens inside `debate.py` and `agents.py` when `run_debate()` is called for one case.

#### 0. Before the debate starts — what every agent already knows

Every agent is instantiated once with a **system message** baked in at construction time (in `agents.py`). The system message is the same for the entire run — it never changes between cases.

Each system message contains:

| Component | Contents |
|-----------|----------|
| **Role description** | Who the agent is and what its job is |
| **Behavioural rules** | What to do, what not to do (e.g. "do NOT produce JSON", "do NOT invent facts") |
| **Issue taxonomy** | The full `ISSUE_TAXONOMY` bullet list from `prompts.py`, injected via `get_issue_guidance()` — this is the shared vocabulary all agents use when naming problems |

The taxonomy injection looks like this at runtime:
```
- inconsistent_work_profile: work-related edits are internally inconsistent or temporally implausible.
- implausible_time_dependent_change: age or education_num changes violate time logic.
- extreme_working_hours: hours_per_week reaches an unrealistic extreme or large jump.
- unactionable_capital_shift: capital_gain or capital_loss changes are financially unrealistic.
- too_many_changes: ...
- fragile_counterfactual: ...
```

#### 1. The opening task message — what the team receives

`_build_case_prompt(case_data, max_rounds)` in `debate.py` constructs the **task message** that is broadcast to the whole `SelectorGroupChat` at the start. Every agent reads this same message.

It contains two layers:

**Layer 1 — Human-readable header** (auto-generated from the case fields):
```
Individual: 42yo Male, Craft-repair, HS-grad, Married-civ-spouse
Model prediction: <=50K (confidence 0.71)
True label: <=50K  |  False negative: False
Number of counterfactuals generated: 4
```

**Layer 2 - Prompt-safe compact case JSON** (the fields needed for evaluation,
pretty-printed):
```json
{
  "case_id": 0,
  "domain": "income_prediction",
  "model_info": { "name": "logistic_regression", "accuracy": 0.8524, ... },
  "original": { "age": 42, "sex": "Male", "occupation": "Craft-repair", ... },
  "prediction": "<=50K",
  "prediction_confidence": 0.71,
  "true_label": "<=50K",
  "is_false_negative": false,
  "counterfactuals": [
    {
      "cf_rank": 0,
      "cf_confidence": 0.63,
      "features_changed": ["occupation", "hours-per-week"],
      "changes_summary": {
        "occupation":     { "from": "Craft-repair",    "to": "Exec-managerial" },
        "hours-per-week": { "from": 40,                "to": 50 }
      }
    },
    { "cf_rank": 1, "cf_confidence": 0.58, ... },
    ...
  ],
  "metrics": {
    "validity": 1.0,
    "continuous_proximity": -0.42,
    "categorical_proximity": 0.91,
    "sparsity": 0.857,
    "continuous_diversity": 0.21,
    "categorical_diversity": 0.18,
    "count_diversity": 0.19
  }
}
```

The prompt deliberately excludes `ground_truth_issues`, `ground_truth_by_cf`,
and `ground_truth_source`. Those labels are used only after the LLM has produced
a verdict, when `run_debate.py` scores the output against the draft reference.
The task message also states the debate structure rules - how many rounds, that
the Judge speaks last, and that agents must not invent evidence.

#### 2. Speaker order — who speaks and when

The `SelectorGroupChat` uses a **selector function** to pick the next speaker after every message. Two modes are available:

**`round_robin` (default):**

```
Turn 1 → Prosecutor
Turn 2 → Defense
Turn 3 → Expert_Witness
Turn 4 → Prosecutor       ← round 2 starts
Turn 5 → Defense
Turn 6 → Expert_Witness
Turn 7 → Judge            ← all specialist rounds exhausted
```

Formula: `specialist_turns = count of specialist messages so far`. Once `specialist_turns ≥ max_rounds × 3`, the Judge is forced next.

**`auto`:**  
The selector narrows the candidate pool at each step (never repeating the same speaker twice in a row), but still forces `Judge` once the round budget is exhausted. The LLM selector prompt then picks from the narrowed pool.

#### 3. What each agent receives when it is selected

In AutoGen's `SelectorGroupChat`, when an agent is selected it receives the **entire conversation history** up to that point — the opening task message plus every message produced by any agent so far. There is no private channel; all agents read all messages.

| When selected | Context the agent sees |
|---------------|------------------------|
| **Prosecutor** (turn 1) | System message + task message (prompt-safe compact case JSON + header + rules) |
| **Defense** (turn 2) | System message + task message + Prosecutor's argument |
| **Expert\_Witness** (turn 3) | System message + task message + Prosecutor's argument + Defense's argument |
| **Prosecutor** (turn 4, round 2) | System message + all previous turns |
| … | … |
| **Judge** (final turn) | System message + task message + **all specialist arguments from all rounds** |

This is the key architectural property: **every agent always has the full debate context**, not just messages directed at them. The Judge in particular synthesises the complete exchange.

#### 4. What each agent is supposed to produce

| Agent | Output format | Output content |
|-------|---------------|----------------|
| **Prosecutor** | Free-form prose | Attacks — specific issues, citing feature values, metrics, issue labels |
| **Defense** | Free-form prose | Counter-arguments — defends actionability, sparsity, diversity |
| **Expert\_Witness** | Free-form prose | Technical reading of DiCE metrics, confidence scores, feasibility assessment |
| **Judge** | Fenced JSON block + `VERDICT_COMPLETE` | Structured verdict (see schema below) |

The Judge's required output format:
````
```json
{
  "case_id": 0,
  "overall_assessment": "fair",
  "flagged_issues": ["low_confidence_cf"],
  "severity": "low",
  "confidence": 0.78,
  "reasoning_summary": "CFs are sparse and actionable but two have barely-above-0.5 confidence.",
  "recommended_action": "review"
}
```
VERDICT_COMPLETE
````

`VERDICT_COMPLETE` on its own line is the termination signal — `TextMentionTermination` watches for it and stops the chat.

#### 5. Verdict parsing and output

After the chat terminates, `utils.parse_judge_verdict()` extracts the JSON from the Judge's last message. It tries three strategies in order:

1. Regex match for a fenced ` ```json ... ``` ` block.
2. Regex match for any fenced ` ``` ... ``` ` block.
3. Balanced-brace scan to extract the first `{...}` object from raw text.

The extracted verdict dict, the full transcript, and cost estimates are returned by `run_debate()` and saved to `results/debate_outputs/`.

#### 6. Full flow diagram

```
results/cases.json
       │
       ▼
run_debate(case_data)
       │
       ├─ _build_case_prompt()  ─────────────────────────────────────────────┐
       │   • human-readable header (age, sex, occupation, prediction, FN flag)│
       │   • prompt-safe compact case JSON (original, counterfactuals[], metrics, model_info) │
       │   • issue taxonomy bullet list                                        │
       │   • debate rules (max_rounds, Judge speaks last)                      │
       └──────────────────────────► SelectorGroupChat.run_stream(task=prompt) │
                                                                               │
  ┌────────────────────────────────────────────────────────────────────────── ┘
  │  All agents share the same model_client and see the full conversation history
  │
  │  ROUND 1
  ├── Prosecutor  ◄── [system_msg] + [task] 
  │   └── attacks: cites cf_confidence, changes_summary, metrics, issue labels
  │
  ├── Defense     ◄── [system_msg] + [task] + [Prosecutor msg]
  │   └── defends: highlights sparsity, actionable features, narrows claims
  │
  ├── Expert_Witness ◄── [system_msg] + [task] + [Prosecutor] + [Defense]
  │   └── technical: reads validity/proximity/sparsity/diversity, flags fragile flips
  │
  │  ROUND 2  (if max_rounds=2)
  ├── Prosecutor  ◄── [system_msg] + [task] + [all round-1 msgs]
  ├── Defense     ◄── [system_msg] + [task] + [all round-1 msgs] + [Prosecutor r2]
  ├── Expert_Witness ◄── all above
  │
  │  FINAL
  └── Judge  ◄── [system_msg] + [task] + [ALL specialist messages]
      └── outputs:  ```json { verdict } ```  VERDICT_COMPLETE
                           │
                    parse_judge_verdict()
                           │
                    save_debate_transcript()   → transcripts/case_XX_transcript.md
                           │
                    <mode>_results.json        → results/debate_outputs/
```

#### 7. Single-LLM baseline — how it differs

When `--single-llm` is passed, `run_single_llm()` is called instead. The flow is simpler:

```
_build_single_llm_prompt(case_data)
  • issue taxonomy
  • instruction to return the Judge JSON schema
  • prompt-safe compact case JSON, without ground-truth labels

       ▼
Single_Evaluator.run(task=prompt)
  • no debate, no other agents
  • reads the case data directly
  • outputs the same JSON schema + VERDICT_COMPLETE

       ▼
parse_judge_verdict()  →  saved to same output structure
```

The single-LLM evaluator uses the same verdict schema as the Judge, making results from both modes directly comparable.

---

## 13. `run_debate.py`

**Purpose** — CLI entry point to run multi-agent debates (or single-LLM baselines) on real pipeline cases.

### Workflow
1. Loads `results/cases.json` (or a custom path via `--cases-file`).
2. Optionally filters to specific case IDs (`--case-ids 0 2 5`).
3. Resolves Groq model/API configuration from CLI args, env vars, or `.env` defaults.
4. For each case, runs either a 4-agent debate (`run_debate`) or a single-LLM evaluation (`--single-llm`).
5. Saves per-run results JSON + per-case Markdown transcripts to `results/debate_outputs/`.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `groq` | Fixed provider; only `groq` is accepted |
| `--model` | `llama-3.1-8b-instant` | Groq model override |
| `--speaker-selection` | `round_robin` | `round_robin` or `auto` |
| `--max-rounds` | `2` | Specialist rounds before Judge |
| `--temperature` | `0.2` | Sampling temperature |
| `--max-tokens` | `700` | Max completion tokens |
| `--single-llm` | off | Run single-LLM baseline instead of debate |
| `--case-ids` | all | Space-separated case IDs to run |
| `--cases-file` | `results/cases.json` | Path to input cases |
| `--delay` | auto | Seconds between cases (rate-limit cooldown) |
| `--verbose` | off | Print agent messages live |

### Example commands
```bash
# Multi-agent debate, all cases, Groq default
$env:PYTHONPATH="src"; python scripts/run_debate.py

# Single-LLM baseline, case 0 only, verbose
$env:PYTHONPATH="src"; python scripts/run_debate.py --single-llm --case-ids 0 --verbose

# Alternative Groq model, auto speaker selection
$env:PYTHONPATH="src"; python scripts/run_debate.py --model llama-3.3-70b-versatile --speaker-selection auto
```

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `results/cases.json`, `.env` (API keys) |
| **Output** | `results/debate_outputs/<model>/<mode>_<timestamp>/` containing `<mode>_results.json` + `transcripts/case_XX_transcript.md` |

### Verdict JSON schema (per case)
```json
{
  "case_id": 0,
  "overall_assessment": "fair | unfair | ambiguous",
  "flagged_issues": ["issue_label_1"],
  "severity": "low | medium | high",
  "confidence": 0.85,
  "reasoning_summary": "...",
  "recommended_action": "accept | review | reject"
}

