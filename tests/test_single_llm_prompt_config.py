import json
import unittest
from pathlib import Path

from agents.agents import SPECIALIST_OUTPUT_PROTOCOL
from agents.config import resolve_llm_config
from agents.debate import (
    _build_case_prompt,
    _build_single_llm_prompt,
    _compact_case_for_prompt,
)
from agents.prompts import get_evidence_guidance, get_issue_guidance


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

    def test_multi_agent_prompt_includes_compact_specialist_protocol(self):
        prompt = _build_case_prompt(self.case, max_rounds=1)

        self.assertIn("Specialist response format for Prosecutor", prompt)
        self.assertIn("ISSUES_SUPPORTED_BY_EVIDENCE:", prompt)
        self.assertIn("ISSUES_NOT_SUPPORTED_OR_OVERSTATED:", prompt)
        self.assertIn("KEY_EVIDENCE:", prompt)
        self.assertIn("BOTTOM_LINE:", prompt)
        self.assertIn("Maximum 90 words total", prompt)
        self.assertIn("Judge response format:", prompt)
        self.assertNotIn("VERDICT_COMPLETE", prompt)

    def test_specialist_protocol_forbids_freeform_json_outputs(self):
        self.assertIn("Do not copy taxonomy definitions", SPECIALIST_OUTPUT_PROTOCOL)
        self.assertIn("Do not produce JSON", SPECIALIST_OUTPUT_PROTOCOL)

    def test_issue_guidance_narrows_work_profile_to_deterministic_evidence(self):
        issue_guidance = get_issue_guidance()
        evidence_guidance = get_evidence_guidance()

        self.assertIn("direct workclass/occupation contradiction", issue_guidance)
        self.assertIn("Taxonomy descriptions define possible labels", evidence_guidance)
        self.assertIn("permitted_range", evidence_guidance)
        self.assertIn("not an actionability guarantee", evidence_guidance)

    def test_non_groq_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_llm_config(provider="gemini")


if __name__ == "__main__":
    unittest.main()
