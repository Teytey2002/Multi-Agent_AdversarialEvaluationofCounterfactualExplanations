"""Feature policy for model training and counterfactual generation.

This module is the single source of truth for the project recourse policy.
The current policy keeps ``education`` out of model training because it is
redundant with ``education-num``, while allowing a limited long-term recourse
regime where age and education level may increase under causal checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


POLICY_NAME = "long_term_actionable_v1"

RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
)

# ``education`` duplicates the ordinal signal in ``education-num`` and adds a
# high-cardinality categorical representation. Keep it in raw artifacts for
# human inspection, but exclude it from model fitting.
MODEL_EXCLUDED_FEATURES: tuple[str, ...] = ("education",)
MODEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for col in RAW_FEATURE_COLUMNS if col not in MODEL_EXCLUDED_FEATURES
)

CONTINUOUS_FEATURES: tuple[str, ...] = (
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)

FEATURE_ALIASES: dict[str, str] = {
    "education-num": "education_num",
    "marital-status": "marital_status",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "hours-per-week": "hours_per_week",
    "native-country": "native_country",
}

ACTIONABLE_FEATURES: tuple[str, ...] = (
    "age",
    "education-num",
    "workclass",
    "occupation",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
)

ACTIONABLE_FEATURES_CANONICAL: tuple[str, ...] = tuple(
    FEATURE_ALIASES.get(feature, feature) for feature in ACTIONABLE_FEATURES
)

FROZEN_FEATURES: tuple[str, ...] = (
    "fnlwgt",
    "marital-status",
    "relationship",
    "race",
    "sex",
    "native-country",
)

FROZEN_FEATURES_CANONICAL: tuple[str, ...] = tuple(
    FEATURE_ALIASES.get(feature, feature) for feature in FROZEN_FEATURES
)

# ``education`` is not an independent mutable feature. It is a human-readable
# display label derived from ``education-num`` after generation.
DERIVED_DISPLAY_FEATURES: tuple[str, ...] = ("education",)

EDUCATION_NUM_TO_LABEL: dict[int, str] = {
    1: "Preschool",
    2: "1st-4th",
    3: "5th-6th",
    4: "7th-8th",
    5: "9th",
    6: "10th",
    7: "11th",
    8: "12th",
    9: "HS-grad",
    10: "Some-college",
    11: "Assoc-voc",
    12: "Assoc-acdm",
    13: "Bachelors",
    14: "Masters",
    15: "Prof-school",
    16: "Doctorate",
}

DICE_DEFAULT_GENETIC_KWARGS: dict[str, Any] = {
    "proximity_weight": 0.2,
    "sparsity_weight": 0.2,
    "diversity_weight": 5.0,
    "categorical_penalty": 0.1,
    "stopping_threshold": 0.5,
    "posthoc_sparsity_param": 0.1,
    "posthoc_sparsity_algorithm": "binary",
}

AGE_MAX_INCREASE = 8
EDUCATION_NUM_MAX_INCREASE = 4
HOURS_MAX_DECREASE = 10
HOURS_MAX_INCREASE = 15
FRAGILITY_THRESHOLD = 0.60
CAPITAL_LARGE_JUMP_THRESHOLD = 3000


def canonical_name(feature: str) -> str:
    """Return the underscore-style feature name used by taxonomy/heuristics."""
    return FEATURE_ALIASES.get(feature, feature)


def _get_feature(row: Mapping[str, Any] | pd.Series, raw: str, canonical: str | None = None) -> Any:
    """Fetch a feature from either raw or canonical column naming."""
    if raw in row:
        return row[raw]
    if canonical and canonical in row:
        return row[canonical]
    return None


def _education_num_level(value: Any) -> int | None:
    """Return a valid Adult education level, accepting integer-like floats."""
    try:
        numeric = float(pd.to_numeric(value))
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric):
        return None

    level = int(round(numeric))
    if abs(numeric - level) > 1e-6:
        return None
    if level not in EDUCATION_NUM_TO_LABEL:
        return None
    return level


def education_label_from_num(value: Any) -> str | None:
    """Map Adult ``education-num`` to its human-readable ``education`` label."""
    level = _education_num_level(value)
    if level is None:
        return None
    return EDUCATION_NUM_TO_LABEL[level]


def sync_education_label(row: pd.Series) -> pd.Series:
    """
    Return a row where ``education`` is synchronized from ``education-num``.

    This keeps generated CF artifacts semantically coherent while preserving the
    modeling decision to exclude ``education`` from classifier inputs.
    """
    out = row.copy()
    education_num = _get_feature(out, "education-num", "education_num")
    label = education_label_from_num(education_num)
    if label is not None and "education" in out:
        out["education"] = label
    return out


def sync_education_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame whose ``education`` labels match ``education-num``."""
    out = df.copy()
    if "education" not in out.columns or "education-num" not in out.columns:
        return out

    labels = out["education-num"].map(education_label_from_num)
    valid = labels.notna()
    out.loc[valid, "education"] = labels.loc[valid]
    return out


def is_synchronized_education_label_change(
    original: Mapping[str, Any] | pd.Series,
    cf: Mapping[str, Any] | pd.Series,
) -> bool:
    """
    Return True when an ``education`` change is only the display counterpart
    of a changed ``education-num`` value.
    """
    if "education" not in original or "education" not in cf:
        return False

    old_label = original["education"]
    new_label = cf["education"]
    if old_label == new_label:
        return False

    old_num = _get_feature(original, "education-num", "education_num")
    new_num = _get_feature(cf, "education-num", "education_num")
    old_level = _education_num_level(old_num)
    new_level = _education_num_level(new_num)
    if old_level is None or new_level is None or old_level == new_level:
        return False

    return str(new_label) == EDUCATION_NUM_TO_LABEL[new_level]


def select_model_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return the feature subset used by the trained classifier."""
    return X.loc[:, list(MODEL_FEATURE_COLUMNS)].copy()


def _safe_float(value: Any) -> float:
    return float(pd.to_numeric(value))


def _quantile(series: pd.Series, q: float) -> float:
    return float(series.dropna().quantile(q))


def _nonzero_quantile(series: pd.Series, q: float, fallback_q: float = 0.95) -> float:
    clean = series.dropna()
    nonzero = clean[clean > 0]
    if len(nonzero) > 0:
        return float(nonzero.quantile(q))
    return float(clean.quantile(fallback_q))


def _bounded_interval(low: float, high: float, include_value: float) -> list[float]:
    """Return an ordered interval that includes the original value."""
    low = min(low, include_value)
    high = max(high, include_value)
    if low > high:
        low, high = high, low
    return [round(float(low), 6), round(float(high), 6)]


def build_permitted_range(data: pd.DataFrame, instance: pd.Series) -> dict[str, list[float]]:
    """
    Build per-instance DiCE box constraints from empirical ranges.

    Dataset percentiles define broad plausible bounds; per-person deltas prevent
    extreme individual jumps. Coupled causal validity, such as education needing
    sufficient age increase, is checked after generation by heuristics.
    """
    age = _safe_float(instance["age"])
    education_num = _safe_float(instance["education-num"])
    hours = _safe_float(instance["hours-per-week"])

    age_max_observed = _quantile(data["age"], 1.0)
    education_max_observed = _quantile(data["education-num"], 1.0)

    hours_min_observed = _quantile(data["hours-per-week"], 0.0)
    hours_max_observed = _quantile(data["hours-per-week"], 1.0)
    hours_p05 = _quantile(data["hours-per-week"], 0.05)
    hours_p95 = _quantile(data["hours-per-week"], 0.95)

    capital_gain_high = _nonzero_quantile(data["capital-gain"], 0.75)
    capital_loss_high = _nonzero_quantile(data["capital-loss"], 0.95)

    age_high = min(age_max_observed, age + AGE_MAX_INCREASE)
    education_high = min(education_max_observed, education_num + EDUCATION_NUM_MAX_INCREASE)

    hours_low = max(hours_min_observed, hours_p05, hours - HOURS_MAX_DECREASE)
    hours_high = min(hours_max_observed, hours_p95, hours + HOURS_MAX_INCREASE)

    return {
        "age": _bounded_interval(age, age_high, age),
        "education-num": _bounded_interval(education_num, education_high, education_num),
        "hours-per-week": _bounded_interval(hours_low, hours_high, hours),
        "capital-gain": [0.0, round(float(capital_gain_high), 6)],
        "capital-loss": [0.0, round(float(capital_loss_high), 6)],
    }


def generation_policy_metadata() -> dict[str, Any]:
    """Return serialisable metadata for generated result artifacts."""
    return {
        "policy_name": POLICY_NAME,
        "model_excluded_features": list(MODEL_EXCLUDED_FEATURES),
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "features_to_vary": list(ACTIONABLE_FEATURES),
        "frozen_features": list(FROZEN_FEATURES),
        "derived_display_features": list(DERIVED_DISPLAY_FEATURES),
        "education_num_to_label": EDUCATION_NUM_TO_LABEL,
        "dice_genetic_kwargs": DICE_DEFAULT_GENETIC_KWARGS,
        "causal_checks": {
            "age_may_only_increase": True,
            "education_num_may_only_increase": True,
            "education_num_increase_requires_age_increase": True,
            "min_age_years_per_education_num_level": 1,
            "education_label_synchronized_from_education_num": True,
        },
    }
