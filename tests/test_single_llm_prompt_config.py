import json
import unittest
from pathlib import Path

from agents.config import resolve_llm_config
from agents.debate import _build_single_llm_prompt, _compact_case_for_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "results" / "cases.json"


class SingleLlmPromptConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CASES_PATH, encoding="utf-8") as f:
            cls.case = json.load(f)[0]

    def test_compact_prompt_payload_excludes_ground_truth_fields(self):
        payload = _compact_case_for_prompt(self.case)

        self.assertNotIn("ground_truth_issues", payload)
        self.assertNotIn("ground_truth_by_cf", payload)
        self.assertNotIn("ground_truth_source", payload)

    def test_single_llm_prompt_excludes_ground_truth_fields(self):
        prompt = _build_single_llm_prompt(self.case)

        self.assertNotIn("ground_truth_issues", prompt)
        self.assertNotIn("ground_truth_by_cf", prompt)
        self.assertNotIn("ground_truth_source", prompt)

    def test_non_groq_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_llm_config(provider="gemini")


if __name__ == "__main__":
    unittest.main()
