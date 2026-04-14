"""
LLM provider configuration — adapted from the AutoGen PoC.

Supports Groq (free tier), Google Gemini, and OpenAI via an OpenAI-compatible
client.  Provider selection and API keys are resolved from CLI arguments,
environment variables, or a local ``.env`` file (looked up from the repo root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root
ENV_PATH = PROJECT_ROOT / ".env"


def load_environment() -> None:
    """Load variables from an optional .env at the repo root."""
    load_dotenv(ENV_PATH, override=False)


# ---------------------------------------------------------------------------
# Defaults & pricing
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "groq"
DEFAULT_MODELS: dict[str, str] = {
    "groq":   "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
}

# Approximate public list prices per 1 M tokens (USD).
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant":                     {"input": 0.05,  "output": 0.08},
    "llama-3.3-70b-versatile":                   {"input": 0.59,  "output": 0.79},
    "gemini-2.5-flash":                          {"input": 0.30,  "output": 2.50},
    "gemini-2.5-flash-lite":                     {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":                              {"input": 0.40,  "output": 1.60},
}

# Groq models need explicit model_info because they aren't recognised by the
# default AutoGen OpenAI client.
GROQ_MODEL_INFO: dict[str, object] = {
    "vision":            False,
    "function_calling":  True,
    "json_output":       True,
    "family":            "unknown",
    "structured_output": False,
}


def _resolve_gemini_family(model_name: str) -> str:
    if model_name.startswith("gemini-2.5-flash"):
        return "gemini-2.5-flash"
    if model_name.startswith("gemini-2.5-pro"):
        return "gemini-2.5-pro"
    if model_name.startswith("gemini-2.0-flash"):
        return "gemini-2.0-flash"
    return "unknown"


# ---------------------------------------------------------------------------
# LLMConfig dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 700
    timeout: float = 90.0
    max_retries: int = 5
    base_url: str | None = None

    @property
    def pricing(self) -> dict[str, float]:
        return MODEL_PRICING_USD_PER_1M.get(self.model, {"input": 0.0, "output": 0.0})


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def resolve_llm_config(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMConfig:
    """Build an ``LLMConfig`` from CLI args + env vars + defaults."""
    load_environment()

    resolved_provider = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).strip().lower()
    if resolved_provider not in {"groq", "gemini", "openai"}:
        raise ValueError(
            f"Unsupported provider '{resolved_provider}'. Use 'groq', 'gemini', or 'openai'."
        )

    resolved_temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))
    resolved_max_tokens  = max_tokens  if max_tokens  is not None else int(os.getenv("LLM_MAX_TOKENS", "700"))

    if resolved_provider == "groq":
        api_key  = os.getenv("GROQ_API_KEY", "")
        resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODELS["groq"])
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        env_var  = "GROQ_API_KEY"
    elif resolved_provider == "gemini":
        api_key  = os.getenv("GEMINI_API_KEY", "")
        resolved_model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODELS["gemini"])
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        env_var  = "GEMINI_API_KEY"
    else:  # openai
        api_key  = os.getenv("OPENAI_API_KEY", "")
        resolved_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"])
        base_url = os.getenv("OPENAI_BASE_URL") or None
        env_var  = "OPENAI_API_KEY"

    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set {env_var} in your environment or in {ENV_PATH}."
        )

    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        base_url=base_url,
    )


def build_model_client(llm_config: LLMConfig) -> OpenAIChatCompletionClient:
    """Create an AutoGen model client from the resolved config."""
    common_kwargs = {
        "model":       llm_config.model,
        "api_key":     llm_config.api_key,
        "temperature": llm_config.temperature,
        "max_tokens":  llm_config.max_tokens,
        "timeout":     llm_config.timeout,
        "max_retries": llm_config.max_retries,
    }

    if llm_config.provider == "groq":
        return OpenAIChatCompletionClient(
            **common_kwargs,
            base_url=llm_config.base_url,
            model_info=GROQ_MODEL_INFO,
            include_name_in_message=False,
            add_name_prefixes=False,
        )

    if llm_config.provider == "gemini":
        return OpenAIChatCompletionClient(
            **common_kwargs,
            base_url=llm_config.base_url,
            model_info={
                "vision":            True,
                "function_calling":  True,
                "json_output":       True,
                "family":            _resolve_gemini_family(llm_config.model),
                "structured_output": False,
            },
            include_name_in_message=False,
            add_name_prefixes=False,
        )

    # OpenAI (native)
    if llm_config.base_url:
        return OpenAIChatCompletionClient(**common_kwargs, base_url=llm_config.base_url)
    return OpenAIChatCompletionClient(**common_kwargs)
