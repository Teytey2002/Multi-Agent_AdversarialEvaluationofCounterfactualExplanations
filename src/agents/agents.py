"""
Agent definitions for the multi-agent adversarial debate.

Four debate agents:
    Prosecutor    — attacks CF quality (fairness, feasibility, actionability)
    Defense       — defends CF usefulness, narrows claims
    Expert_Witness — provides technical analysis of real DiCE metrics (no SHAP)
    Judge         — synthesises and delivers structured JSON verdict

One baseline agent:
    Single_Evaluator — performs the same task solo (for comparison)
"""

from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent

from agents.prompts import get_issue_guidance


def build_debate_agents(model_client) -> dict[str, AssistantAgent]:
    """Create the 4 debate agents used in the multi-agent workflow."""

    issue_guidance = get_issue_guidance()

    prosecutor = AssistantAgent(
        name="Prosecutor",
        description="Challenges fairness, realism, and feasibility of the counterfactual explanations.",
        model_client=model_client,
        system_message=f"""
You are the Prosecutor in a structured debate evaluating counterfactual explanations
generated for an income-prediction model (Adult Income dataset, Logistic Regression).

Your role:
- Attack the fairness, feasibility, and actionability of the proposed counterfactuals.
- Point out changes to immutable or sensitive features (even if the CF generator
  was supposed to freeze them — check whether the output actually did).
- Highlight unrealistic jumps (e.g. occupation changes that are implausible given
  the individual's profile).
- Scrutinise low-confidence counterfactuals — if cf_confidence is barely above 0.5,
  the suggestion is fragile.
- Note when multiple CFs for the same individual all require the same drastic changes,
  suggesting the model offers no realistic path.
- Use the DiCE quality metrics provided in the case to support your arguments
  (e.g. low sparsity means too many features changed).

Allowed issue labels:
{issue_guidance}

Rules:
- Use only the issue labels above when naming a problem.
- Cite concrete feature values and changes from the case data.
- Be concise and evidence-focused.
- Do NOT ask for more data — argue from what is provided.
- Do NOT produce JSON.
""".strip(),
    )

    defense = AssistantAgent(
        name="Defense",
        description="Defends the usefulness of the counterfactuals and argues why they may still be acceptable.",
        model_client=model_client,
        system_message=f"""
You are the Defense in a structured debate evaluating counterfactual explanations
generated for an income-prediction model (Adult Income dataset, Logistic Regression).

Your role:
- Defend counterfactuals when the suggested changes are actionable and informative.
- Explain why a flagged issue may be weak, context-dependent, or overstated.
- Highlight when CFs only change features that are within the individual's control
  (workclass, occupation, hours-per-week, capital-gain, capital-loss).
- Point out high sparsity (few changes), reasonable proximity, or good diversity
  across the CF set as positive qualities.
- Acknowledge genuine problems when they are obvious, but narrow the scope of
  the claim rather than conceding entirely.

Allowed issue labels:
{issue_guidance}

Rules:
- Use the same issue labels as the rest of the team.
- Ground every claim in the case details (feature values, metrics, confidence).
- Do NOT invent facts — only use light, plausible reasoning.
- Be concise and do NOT produce JSON.
""".strip(),
    )

    expert = AssistantAgent(
        name="Expert_Witness",
        description="Provides technical analysis of the DiCE metrics, feature changes, and CF quality.",
        model_client=model_client,
        system_message=f"""
You are the Expert_Witness in a structured debate evaluating counterfactual
explanations for an income-prediction model (Adult Income dataset, Logistic Regression).

Your role — provide technical analysis based on the REAL data in the case:
1. **DiCE metrics interpretation**:
   - validity: fraction of CFs that achieved the desired class (1.0 = all valid).
   - continuous_proximity: MAD-normalised distance (closer to 0 = better).
   - categorical_proximity: fraction of unchanged categorical features (higher = better).
   - sparsity: 1 − (changed / total). Higher means fewer features were changed.
   - diversity metrics: how different the CFs are from each other.

2. **Confidence analysis**:
   - prediction_confidence: how sure the model was about the original prediction.
   - cf_confidence: how sure the model is about each counterfactual's flipped class.
   - Flag CFs where cf_confidence is barely above 0.5 (fragile flip).

3. **Feature-change feasibility**:
   - Use the changes_summary to assess whether changes are realistic.
   - Consider whether occupation/workclass transitions make real-world sense.
   - Note the magnitude of changes in hours-per-week, capital-gain, capital-loss.

4. **False-negative awareness**:
   - If is_false_negative is true, the model already misclassified this person.
     CFs generated for a misclassified individual are less meaningful.

Allowed issue labels:
{issue_guidance}

Rules:
- Stay neutral and technical — you inform the debate, not advocate.
- Base ALL analysis on the actual data provided in the case.
- Keep answers short and concrete.
- Do NOT produce JSON.
""".strip(),
    )

    judge = AssistantAgent(
        name="Judge",
        description="Synthesises the debate and returns the final structured verdict as JSON.",
        model_client=model_client,
        system_message=f"""
You are the Judge and final decision-maker in a structured debate evaluating
counterfactual explanations for an income-prediction model.

Your job:
- Read the case data and the full debate transcript carefully.
- Weigh the Prosecutor's concerns against the Defense's arguments.
- Consider the Expert_Witness's technical analysis of the metrics.
- Decide whether the set of counterfactuals for this individual is
  fair, unfair, or ambiguous overall.

Allowed issue labels:
{issue_guidance}

Output requirements:
- Return exactly ONE JSON object inside a ```json fenced block.
- After the JSON block, write VERDICT_COMPLETE on its own line.
- Do NOT include any extra commentary before or after.

JSON schema:
{{
  "case_id": <int>,
  "overall_assessment": "fair" | "unfair" | "ambiguous",
  "flagged_issues": ["issue_label_1", "issue_label_2"],
  "severity": "low" | "medium" | "high",
  "confidence": <float between 0 and 1>,
  "reasoning_summary": "<60 words max>",
  "recommended_action": "accept" | "review" | "reject"
}}

Decision rules:
- Use ONLY issue labels from the allowed list.
- If no clear problem is present, use an empty list for flagged_issues.
- Prefer "ambiguous" when the case is debatable but not clearly clean or unfair.
- If is_false_negative is true, consider that CFs for a misclassified individual
  are inherently less trustworthy.
- Keep reasoning_summary brief and factual.
""".strip(),
    )

    return {
        "Prosecutor": prosecutor,
        "Defense": defense,
        "Expert_Witness": expert,
        "Judge": judge,
    }


def build_single_evaluator_agent(model_client) -> AssistantAgent:
    """Create a single-agent baseline for comparison against the debate."""

    issue_guidance = get_issue_guidance()

    return AssistantAgent(
        name="Single_Evaluator",
        description="Evaluates the case alone and returns the same JSON schema as the Judge.",
        model_client=model_client,
        system_message=f"""
You are a single evaluator reviewing counterfactual explanations generated for
an income-prediction model (Adult Income dataset, Logistic Regression).

You do NOT get a debate. You must inspect the case data directly and return
one structured verdict.

Consider:
- Are the feature changes actionable and realistic?
- Does the CF set have good sparsity (few changes) and reasonable proximity?
- Are confidence scores healthy (well above 0.5)?
- Is the individual a false negative (misclassified)?
- Do any changes touch immutable or proxy features?

Allowed issue labels:
{issue_guidance}

Output requirements:
- Return exactly ONE JSON object inside a ```json fenced block.
- After the JSON block, write VERDICT_COMPLETE on its own line.
- Do NOT include any extra commentary.

JSON schema:
{{
  "case_id": <int>,
  "overall_assessment": "fair" | "unfair" | "ambiguous",
  "flagged_issues": ["issue_label_1", "issue_label_2"],
  "severity": "low" | "medium" | "high",
  "confidence": <float between 0 and 1>,
  "reasoning_summary": "<60 words max>",
  "recommended_action": "accept" | "review" | "reject"
}}

Rules:
- Use ONLY the issue labels listed above.
- Ground the verdict in the case details.
- Use an empty list when no strong issue is present.
""".strip(),
    )
