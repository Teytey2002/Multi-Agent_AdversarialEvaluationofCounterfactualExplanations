"""
Groq LLM configuration.

The project standardises on Groq because it is the only free provider used in
the experiment series with sufficiently generous limits for repeated case runs.
The Groq API is consumed through AutoGen's OpenAI-compatible client.
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
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_MODELS: dict[str, str] = {"groq": DEFAULT_GROQ_MODEL}

# Approximate public list prices per 1 M tokens (USD).
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant":                     {"input": 0.05,  "output": 0.08},
    "llama-3.3-70b-versatile":                   {"input": 0.59,  "output": 0.79},
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

    requested_provider = (provider or DEFAULT_PROVIDER).strip().lower()
    if requested_provider != "groq":
        raise ValueError("This project is configured for Groq only. Use provider='groq'.")

    resolved_temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))
    resolved_max_tokens  = max_tokens  if max_tokens  is not None else int(os.getenv("LLM_MAX_TOKENS", "700"))

    api_key  = os.getenv("GROQ_API_KEY", "")
    resolved_model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set GROQ_API_KEY in your environment or in {ENV_PATH}."
        )

    return LLMConfig(
        provider="groq",
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

    return OpenAIChatCompletionClient(
        **common_kwargs,
        base_url=llm_config.base_url,
        model_info=GROQ_MODEL_INFO,
        include_name_in_message=False,
        add_name_prefixes=False,
    )
