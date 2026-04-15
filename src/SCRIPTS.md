# Pipeline Scripts — Per-Script Documentation

> Quick reference for every script in `src/`.
> Run order: **data_loader → explore_data → preprocessing → models → train → predict → generate_cf → cf_metrics → case_builder → run_debate**
> (utils is a shared helper module; agents/ is a package used by run_debate.)

All scripts must be executed from the **repo root** with `PYTHONPATH=src`:

```bash
# PowerShell
$env:PYTHONPATH="src"; python src/<script>.py

# Bash / Linux
PYTHONPATH=src python src/<script>.py
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
$env:PYTHONPATH="src"; python src/explore_data.py
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
   - **Actionable features only** (workclass, occupation, hours-per-week, capital-gain, capital-loss).
   - **Box constraints** on continuous features to avoid absurd values.
   - Post-hoc sparsity via `posthoc_sparsity_param=0.1`.
6. Saves a structured CSV with `row_type` (original / counterfactual), `original_index`, `cf_rank`, and `cf_confidence` (model's P(class 1) for each row) columns.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `TOTAL_CFS` | `4` | Number of CFs per instance |
| `DESIRED_CLASS` | `1` | Target: >50 K income |
| `method` | `"genetic"` | Best available for sklearn backend |
| `proximity_weight` | `1.0` | Trade-off: keep CFs close to original |
| `diversity_weight` | `3.0` | Trade-off: diversify the set of CFs |
| `categorical_penalty` | `1.0` | Penalty for categorical feature changes |
| `stopping_threshold` | `0.5` | Probability threshold to accept a CF |
| `posthoc_sparsity_param` | `0.1` | Linear sparsity post-processing |

#### Actionable features
| Feature | Rationale |
|---------|-----------|
| `workclass` | Job sector can change |
| `occupation` | Job role can change |
| `hours-per-week` | Working hours adjustable |
| `capital-gain` | Financial gains can vary |
| `capital-loss` | Financial losses can vary |

Protected / immutable features (age, race, sex, native-country, education, marital-status, relationship, fnlwgt, education-num) are **frozen**.

#### Box constraints
| Feature | Min | Max |
|---------|-----|-----|
| `hours-per-week` | 20 | 50 |
| `capital-gain` | 0 | 5 000 |
| `capital-loss` | 0 | 5 000 |

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
1. Reads four pipeline artifacts:
   - `results/unfavorable_samples.csv` — the 10 sampled individuals.
   - `results/counterfactuals.csv` — originals + counterfactual rows (row_type / cf_rank).
   - `results/cf_metrics.csv` — per-instance DiCE quality metrics.
   - `results/logistic_regression_metrics.json` — model evaluation metrics.
2. For each sample, locates its CFs via `original_index`.
3. Computes `changes_summary` per CF: `{feature: {from: ..., to: ...}}` for every changed feature.
4. Assembles a case dict with:
   - Person profile (`original`), model prediction + confidence, true label, false-negative flag.
   - `counterfactuals` array (one entry per CF, each with features, features_changed, changes_summary, cf_confidence).
   - `metrics` block (validity, proximity, sparsity, diversity).
   - `model_info` (name, accuracy, precision, recall, F1).
   - `ground_truth_issues` (empty — placeholder for Ivan's issue taxonomy).
5. Writes all cases to a single JSON file.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `--out` | `results/cases.json` | Output path (default) |
| `--pretty` | flag | Indent JSON for readability |

### CLI
```bash
$env:PYTHONPATH="src"; python src/case_builder.py --pretty
```

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `results/unfavorable_samples.csv`, `results/counterfactuals.csv`, `results/cf_metrics.csv`, `results/logistic_regression_metrics.json` |
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
  "ground_truth_issues": []
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

## 10. `agents/` package

**Purpose** — Multi-agent adversarial debate system for evaluating counterfactual explanations, adapted from the AutoGen PoC.

### Package structure

| Module | Description |
|--------|-------------|
| `__init__.py` | Package marker; re-exports `build_debate_agents`, `build_single_evaluator_agent`, `run_debate`, `run_single_llm` |
| `prompts.py` | **Issue taxonomy** (placeholder — Ivan replaces this) + `get_issue_guidance()` formatter |
| `agents.py` | 4 debate agents (Prosecutor, Defense, Expert\_Witness, Judge) + 1 single-LLM baseline (Single\_Evaluator) |
| `config.py` | `LLMConfig` dataclass, provider resolution (Groq / Gemini / OpenAI), `build_model_client()` |
| `debate.py` | `run_debate()` / `run_single_llm()` — orchestrates `SelectorGroupChat` with round-robin or auto speaker selection |
| `utils.py` | `parse_judge_verdict()`, `serialise_message()`, `calculate_cost()`, `save_debate_transcript()`, `compute_agreement()` |

### Agent roles

| Agent | Role |
|-------|------|
| **Prosecutor** | Attacks CF quality — fairness, feasibility, actionability, low confidence |
| **Defense** | Defends useful CFs, narrows claims, highlights actionable features |
| **Expert\_Witness** | Technical analysis of real DiCE metrics, confidence scores, feature-change feasibility (**no SHAP**) |
| **Judge** | Synthesises debate → structured JSON verdict (`fair` / `unfair` / `ambiguous`) |
| **Single\_Evaluator** | Same task as Judge but without a debate (baseline comparison) |

### Key design decisions (vs PoC)
- **Multi-CF schema**: prompts handle `counterfactuals[]` array with per-CF `cf_confidence`, `features_changed`, `changes_summary`.
- **Expert Witness — no SHAP**: analyses real DiCE metrics (validity, proximity, sparsity, diversity) + domain feasibility instead of mocked SHAP values.
- **Placeholder taxonomy**: `prompts.py` contains a default 7-label taxonomy; Ivan's finalised taxonomy will replace it.
- **Real data**: loads `results/cases.json` (from `case_builder.py`), not mock cases.

### Environment setup
Requires a `.env` file at the repo root with at least one API key:

```
GROQ_API_KEY=gsk_...
# or
GEMINI_API_KEY=AIza...
# or
OPENAI_API_KEY=sk-...
```

---

## 11. `run_debate.py`

**Purpose** — CLI entry point to run multi-agent debates (or single-LLM baselines) on real pipeline cases.

### Workflow
1. Loads `results/cases.json` (or a custom path via `--cases-file`).
2. Optionally filters to specific case IDs (`--case-ids 0 2 5`).
3. Resolves LLM provider/model from CLI args, env vars, or `.env` defaults.
4. For each case, runs either a 4-agent debate (`run_debate`) or a single-LLM evaluation (`--single-llm`).
5. Saves per-run results JSON + per-case Markdown transcripts to `results/debate_outputs/`.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `groq` | LLM provider: `groq`, `gemini`, `openai` |
| `--model` | per-provider | Model override |
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
$env:PYTHONPATH="src"; python src/run_debate.py

# Single-LLM baseline, case 0 only, verbose
$env:PYTHONPATH="src"; python src/run_debate.py --single-llm --case-ids 0 --verbose

# Gemini provider, auto speaker selection
$env:PYTHONPATH="src"; python src/run_debate.py --provider gemini --speaker-selection auto
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
