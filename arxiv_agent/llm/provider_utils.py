"""Shared helpers for LLM provider selection and credential lookup."""

from __future__ import annotations

import os

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def get_provider_api_key(provider: str) -> str | None:
    """Return the configured API key for a supported remote provider."""
    env_var = PROVIDER_ENV_VARS.get(provider.lower())
    if not env_var:
        return None
    return os.getenv(env_var)


def get_provider_env_var(provider: str) -> str | None:
    """Return the environment variable name for a provider."""
    return PROVIDER_ENV_VARS.get(provider.lower())
