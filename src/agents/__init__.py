"""
agents — Multi-agent adversarial debate system for counterfactual evaluation.

Public API
----------
- build_debate_agents(model_client) → dict of 4 debate agents
- build_single_evaluator_agent(model_client) → single-agent baseline
- run_debate(case_data, **kwargs)   → synchronous debate runner
- run_single_llm(case_data, **kwargs) → synchronous single-LLM runner
"""

from agents.agents import build_debate_agents, build_single_evaluator_agent  # noqa: F401
from agents.debate import run_debate, run_single_llm                        # noqa: F401
