import json
import unittest
from pathlib import Path

from agents.prompts import get_valid_issue_labels


ANNOTATIONS_PATH = Path("annotations") / "ground_truth_labels.json"


class GroundTruthAnnotationTests(unittest.TestCase):
    def test_ground_truth_annotations_use_valid_taxonomy_labels(self):
        data = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        valid_labels = get_valid_issue_labels()

        self.assertEqual(set(str(i) for i in range(10)), set(data["cases"]))

        for case_id, case_data in data["cases"].items():
            case_labels = set(case_data["ground_truth_issues"])
            self.assertTrue(
                case_labels.issubset(valid_labels),
                f"case {case_id} has invalid labels: {case_labels - valid_labels}",
            )

            cf_union = set()
            for cf_labels in case_data["ground_truth_by_cf"].values():
                cf_label_set = set(cf_labels)
                self.assertTrue(
                    cf_label_set.issubset(valid_labels),
                    f"case {case_id} has invalid per-CF labels: {cf_label_set - valid_labels}",
                )
                cf_union.update(cf_label_set)

            self.assertEqual(
                case_labels,
                cf_union,
                f"case {case_id} case-level labels should equal the per-CF label union",
            )


if __name__ == "__main__":
    unittest.main()
