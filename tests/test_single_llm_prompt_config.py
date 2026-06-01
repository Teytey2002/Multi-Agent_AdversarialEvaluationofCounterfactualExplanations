import json
import unittest
from pathlib import Path

from agents.agents import (
    SPECIALIST_OUTPUT_PROTOCOL,
    SINGLE_EXPLANATION_LAYER_GUIDANCE,
    SINGLE_EVALUATOR_PHASE2_CALIBRATION,
    _build_single_evaluator_system_message,
)
from agents.config import resolve_llm_config
from agents.debate import (
    _build_case_prompt,
    _build_single_explanation_prompt,
    _build_single_llm_prompt,
    _compact_case_for_prompt,
)
from agents.prompts import get_evidence_guidance, get_issue_guidance
from agents.utils import parse_judge_verdict


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

    def test_issue_guidance_uses_phase_2_five_label_taxonomy(self):
        issue_guidance = get_issue_guidance()
        evidence_guidance = get_evidence_guidance()
        expected_labels = {
            "extreme_working_hours",
            "implausible_time_dependent_change",
            "unactionable_capital_shift",
            "too_many_changes",
            "fragile_counterfactual",
        }

        for label in expected_labels:
            self.assertIn(label, issue_guidance)
        self.assertNotIn("inconsistent_work_profile", issue_guidance)
        self.assertIn("Taxonomy descriptions define possible labels", evidence_guidance)
        self.assertIn("permitted_range", evidence_guidance)
        self.assertIn("not an actionability guarantee", evidence_guidance)

    def test_single_evaluator_has_phase_2_substitution_calibration(self):
        system_message = _build_single_evaluator_system_message()

        self.assertIn("Phase 2 substitution calibration", system_message)
        self.assertIn("metrics-only reference system", system_message)
        self.assertIn("heuristic_summary.flagged_issues_union", system_message)
        self.assertIn("Include every candidate issue", system_message)
        self.assertIn("absent from both", system_message)
        self.assertIn("fragile_counterfactual", system_message)
        self.assertIn("Do not drop `fragile_counterfactual`", system_message)
        self.assertIn("cf_confidence", system_message)
        self.assertIn(
            "not to invent a new evaluation policy",
            SINGLE_EVALUATOR_PHASE2_CALIBRATION,
        )

    def test_explainability_layer_is_decoupled_from_verdict_prompt(self):
        system_message = _build_single_evaluator_system_message()
        self.assertNotIn("expert_explanation", system_message)

        fixed_verdict = {
            "case_id": self.case["case_id"],
            "overall_assessment": "unfair",
            "flagged_issues": ["fragile_counterfactual"],
            "severity": "medium",
            "confidence": 0.86,
            "reasoning_summary": "Fragility is supported by heuristic evidence.",
            "recommended_action": "review",
        }
        explanation_prompt = _build_single_explanation_prompt(
            self.case,
            fixed_verdict,
        )

        self.assertIn("fixed single-LLM evaluation", explanation_prompt)
        self.assertIn("expert_explanation", explanation_prompt)
        self.assertIn("Do not revise the fixed evaluation", explanation_prompt)
        self.assertIn("counterfactual explanation set", explanation_prompt)
        self.assertIn("issue_alignment", explanation_prompt)
        self.assertIn("fragile_counterfactual", explanation_prompt)
        self.assertNotIn("ground_truth_issues", explanation_prompt)
        self.assertIn("do not write prose outside", SINGLE_EXPLANATION_LAYER_GUIDANCE)
        self.assertIn("Never say the original model prediction", SINGLE_EXPLANATION_LAYER_GUIDANCE)

    def test_verdict_parser_preserves_expert_explanation_field(self):
        verdict = parse_judge_verdict(
            """
            ```json
            {
              "case_id": 7,
              "overall_assessment": "unfair",
              "flagged_issues": ["fragile_counterfactual"],
              "severity": "medium",
              "confidence": 0.86,
              "expert_explanation": "The counterfactuals barely pass the model threshold, so the advice is unstable.",
              "reasoning_summary": "Fragility is supported by heuristic evidence.",
              "recommended_action": "review"
            }
            ```
            VERDICT_COMPLETE
            """
        )

        self.assertEqual(
            verdict["expert_explanation"],
            "The counterfactuals barely pass the model threshold, so the advice is unstable.",
        )
        self.assertEqual(verdict["flagged_issues"], ["fragile_counterfactual"])

    def test_non_groq_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_llm_config(provider="gemini")


if __name__ == "__main__":
    unittest.main()
