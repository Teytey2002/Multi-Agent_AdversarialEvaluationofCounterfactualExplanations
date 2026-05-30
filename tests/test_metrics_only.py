import unittest

from evaluators.metrics_only import evaluate_case_metrics_only


def _base_case(**overrides):
    case = {
        "case_id": 0,
        "is_false_negative": False,
        "metrics": {
            "validity": 1.0,
            "sparsity": 0.85,
            "continuous_proximity": -0.2,
            "categorical_proximity": 0.9,
            "count_diversity": 0.5,
        },
        "heuristic_summary": {
            "flagged_issues_union": [],
            "constraint_violations_union": [],
        },
        "counterfactuals": [],
        "ground_truth_issues": [],
    }
    case.update(overrides)
    return case


class MetricsOnlyEvaluatorTests(unittest.TestCase):
    def test_clean_case_is_accepted(self):
        verdict = evaluate_case_metrics_only(_base_case())

        self.assertEqual(verdict["overall_assessment"], "fair")
        self.assertEqual(verdict["recommended_action"], "accept")
        self.assertEqual(verdict["flagged_issues"], [])

    def test_fragile_only_case_is_ambiguous_review(self):
        case = _base_case(
            heuristic_summary={
                "flagged_issues_union": ["fragile_counterfactual"],
                "constraint_violations_union": [],
            }
        )

        verdict = evaluate_case_metrics_only(case)

        self.assertEqual(verdict["overall_assessment"], "ambiguous")
        self.assertEqual(verdict["severity"], "low")
        self.assertEqual(verdict["recommended_action"], "review")
        self.assertEqual(verdict["flagged_issues"], ["fragile_counterfactual"])

    def test_multiple_critical_issues_are_rejected(self):
        case = _base_case(
            heuristic_summary={
                "flagged_issues_union": ["too_many_changes", "unactionable_capital_shift"],
                "constraint_violations_union": [],
            }
        )

        verdict = evaluate_case_metrics_only(case)

        self.assertEqual(verdict["overall_assessment"], "unfair")
        self.assertEqual(verdict["severity"], "high")
        self.assertEqual(verdict["recommended_action"], "reject")

    def test_constraint_violation_is_not_added_to_flagged_issues(self):
        case = _base_case(
            heuristic_summary={
                "flagged_issues_union": [],
                "constraint_violations_union": ["sex_changed_despite_being_frozen"],
            }
        )

        verdict = evaluate_case_metrics_only(case)

        self.assertEqual(verdict["flagged_issues"], [])
        self.assertIn("constraint_violations", verdict)
        self.assertEqual(verdict["overall_assessment"], "unfair")

    def test_falls_back_to_counterfactual_level_issues(self):
        case = _base_case(
            heuristic_summary={
                "flagged_issues_union": [],
                "constraint_violations_union": [],
            },
            counterfactuals=[
                {
                    "heuristic_metrics": {
                        "flagged_issues": ["extreme_working_hours"],
                        "constraint_violations": [],
                    }
                }
            ],
        )

        verdict = evaluate_case_metrics_only(case)

        self.assertEqual(verdict["flagged_issues"], ["extreme_working_hours"])
        self.assertEqual(verdict["severity"], "medium")

    def test_low_count_diversity_is_metric_warning_not_scored_issue(self):
        case = _base_case(
            metrics={
                "validity": 1.0,
                "sparsity": 0.85,
                "continuous_proximity": -0.2,
                "categorical_proximity": 0.9,
                "count_diversity": 0.1,
            }
        )

        verdict = evaluate_case_metrics_only(case)

        self.assertEqual(verdict["flagged_issues"], [])
        self.assertEqual(verdict["metric_warnings"], ["low_count_diversity"])
        self.assertEqual(verdict["overall_assessment"], "ambiguous")

if __name__ == "__main__":
    unittest.main()
