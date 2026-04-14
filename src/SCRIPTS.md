# Pipeline Scripts — Per-Script Documentation

> Quick reference for every script in `src/`.
> Run order: **data_loader → preprocessing → models → train → predict → generate_cf → cf_metrics**
> (utils is a shared helper module, not run directly.)

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
3. Filters *unfavorable* cases (predicted class = 0, i.e. ≤50 K).
4. Randomly samples 5 unfavorable individuals (`random_state=42`).
5. Saves the sample to CSV.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_PATH` | `"models/logistic_regression.joblib"` | Trained pipeline |
| Sample size | `5` | Number of unfavorable individuals to pick |
| `random_state` | `42` | Reproducible sampling |

### Inputs / Outputs
| | Description |
|---|---|
| **Input** | `models/logistic_regression.joblib`, OpenML Adult dataset |
| **Output** | `results/unfavorable_samples.csv` (5 rows with features + prediction + proba + true_label) |

### Note
Some sampled rows may have NaN in `workclass` / `occupation` (original dataset missing values encoded as `?`). These are silently dropped in the next step (`generate_cf.py`).

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
6. Saves a structured CSV with `row_type` (original / counterfactual), `original_index`, and `cf_rank` columns.

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
| **Output** | `results/counterfactuals.csv` — structured table with originals + their CFs |

---

## 7. `cf_metrics.py`

**Purpose** — Evaluate counterfactual quality using DiCE-paper metrics (MAD normalization).

### Workflow
1. Loads the full Adult dataset to compute feature types and MAD (Median Absolute Deviation) values.
2. Reads `results/counterfactuals.csv`.
3. Groups rows by `original_index`, separates originals from counterfactuals.
4. For each instance, computes:
   - **Validity** — fraction of unique CFs that achieved the desired class.
   - **Continuous proximity** — average MAD-normalized distance (negative = closer is better).
   - **Categorical proximity** — average fraction of unchanged categorical features.
   - **Sparsity** — `1 − (changed_features / total_features)`. Higher = sparser.
   - **Continuous diversity** — pairwise MAD-normalized distance between CFs.
   - **Categorical diversity** — pairwise fraction of differing categorical features.
   - **Count diversity** — pairwise fraction of differing features (any type).
5. Saves per-instance and global (averaged) metrics.

### Key parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| MAD normalization | Per-column MAD from full dataset | Falls back to 1.0 if MAD = 0 |
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
