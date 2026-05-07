import unittest

import pandas as pd

from agents.prompts import get_valid_issue_labels
from feature_policy import (
    DICE_DEFAULT_GENETIC_KWARGS,
    EDUCATION_NUM_TO_LABEL,
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    build_permitted_range,
    is_synchronized_education_label_change,
    select_model_features,
    sync_education_labels,
)
from heuristics import compute_heuristic_metrics


def _adult_frame() -> pd.DataFrame:
    row = {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 120000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Sales",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    older = {
        **row,
        "age": 60,
        "education-num": 13,
        "capital-gain": 5000,
        "capital-loss": 2000,
        "hours-per-week": 55,
    }
    return pd.DataFrame([row, older], columns=list(RAW_FEATURE_COLUMNS))


class FoundationPolicyTests(unittest.TestCase):
    def test_model_feature_selection_excludes_education_only(self):
        X = _adult_frame()
        selected = select_model_features(X)

        self.assertNotIn("education", selected.columns)
        self.assertIn("education-num", selected.columns)
        self.assertEqual(list(selected.columns), list(MODEL_FEATURE_COLUMNS))

    def test_dice_genetic_parameters_match_reference_defaults(self):
        self.assertEqual(DICE_DEFAULT_GENETIC_KWARGS["proximity_weight"], 0.2)
        self.assertEqual(DICE_DEFAULT_GENETIC_KWARGS["sparsity_weight"], 0.2)
        self.assertEqual(DICE_DEFAULT_GENETIC_KWARGS["diversity_weight"], 5.0)
        self.assertEqual(DICE_DEFAULT_GENETIC_KWARGS["categorical_penalty"], 0.1)
        self.assertEqual(DICE_DEFAULT_GENETIC_KWARGS["stopping_threshold"], 0.5)

    def test_permitted_range_is_empirical_and_non_decreasing_for_time_features(self):
        X = _adult_frame()
        ranges = build_permitted_range(X, X.iloc[0])

        self.assertEqual(ranges["age"][0], 35.0)
        self.assertGreaterEqual(ranges["age"][1], ranges["age"][0])
        self.assertEqual(ranges["education-num"][0], 9.0)
        self.assertGreaterEqual(ranges["education-num"][1], ranges["education-num"][0])
        self.assertIn("hours-per-week", ranges)
        self.assertIn("capital-gain", ranges)

    def test_time_dependent_change_is_scored_issue_not_frozen_feature(self):
        original = {
            "age": 35,
            "education_num": 9,
            "hours_per_week": 40,
            "workclass": "Private",
            "occupation": "Sales",
            "sex": "Female",
        }
        cf = {**original, "education_num": 12}

        metrics = compute_heuristic_metrics(original, cf, cf_confidence=0.7)

        self.assertIn("implausible_time_dependent_change", metrics["flagged_issues"])
        self.assertNotIn(
            "education_num_changed_despite_being_frozen",
            metrics["constraint_violations"],
        )

    def test_frozen_feature_change_remains_constraint_violation(self):
        original = {
            "age": 35,
            "education_num": 9,
            "hours_per_week": 40,
            "sex": "Female",
        }
        cf = {**original, "sex": "Male"}

        metrics = compute_heuristic_metrics(original, cf, cf_confidence=0.7)

        self.assertEqual(metrics["flagged_issues"], [])
        self.assertIn("sex_changed_despite_being_frozen", metrics["constraint_violations"])

    def test_new_taxonomy_label_is_valid_for_judge_verdicts(self):
        self.assertIn("implausible_time_dependent_change", get_valid_issue_labels())

    def test_education_label_is_synchronized_from_education_num(self):
        cf_df = pd.DataFrame([
            {
                "education": "HS-grad",
                "education-num": 13,
            }
        ])

        synced = sync_education_labels(cf_df)

        self.assertEqual(EDUCATION_NUM_TO_LABEL[13], "Bachelors")
        self.assertEqual(synced.loc[0, "education"], "Bachelors")

    def test_synchronized_education_label_change_is_not_independent_change(self):
        original = {
            "age": 35,
            "education": "HS-grad",
            "education_num": 9,
            "hours_per_week": 40,
        }
        cf = {
            **original,
            "age": 39,
            "education": "Bachelors",
            "education_num": 13,
        }

        metrics = compute_heuristic_metrics(original, cf, cf_confidence=0.7)

        self.assertTrue(is_synchronized_education_label_change(original, cf))
        self.assertIn("education_num", metrics["changed_features"])
        self.assertNotIn("education", metrics["changed_features"])
        self.assertNotIn(
            "education_changed_without_education_num_sync",
            metrics["constraint_violations"],
        )

    def test_unsynchronized_education_label_change_is_constraint_violation(self):
        original = {
            "age": 35,
            "education": "HS-grad",
            "education_num": 9,
            "hours_per_week": 40,
        }
        cf = {
            **original,
            "education": "Bachelors",
            "education_num": 9,
        }

        metrics = compute_heuristic_metrics(original, cf, cf_confidence=0.7)

        self.assertIn("education", metrics["changed_features"])
        self.assertIn(
            "education_changed_without_education_num_sync",
            metrics["constraint_violations"],
        )


if __name__ == "__main__":
    unittest.main()
