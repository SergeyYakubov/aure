"""
LLM provider registry.

Supported providers: openai, gemini, alcf, local.

``openai``, ``alcf``, and ``local`` all use the OpenAI-compatible
LangChain wrapper with different base URLs and credential handling.
``gemini`` uses :class:`ChatGoogleGenerativeAI`.

Adding a new OpenAI-compatible provider only requires a new factory in
``openai_compat.py`` and a registry entry in :data:`PROVIDERS` below.
"""

from __future__ import annotations

from typing import Optional

from ..config import get_llm_config
from .alcf_auth import get_token  # noqa: F401  (re-exported for config.py)
from .openai_compat import create_openai, create_alcf, create_local
from .gemini import create_gemini

# ── Registry & public entry point ──────────────────────────────────────

PROVIDERS = {
    "openai": create_openai,
    "gemini": create_gemini,
    "alcf": create_alcf,
    "local": create_local,
}


def get_llm(temperature: Optional[float] = None):
    """Return a configured LangChain chat model for the active provider.

    Args:
        temperature: Override the configured temperature.

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured.
    """
    config = get_llm_config()
    provider = config["provider"]

    factory = PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(sorted(PROVIDERS))}"
        )

    temp = temperature if temperature is not None else config["temperature"]
    return factory(config, temp)
