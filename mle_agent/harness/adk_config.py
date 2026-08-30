"""Google ADK model configuration for the autonomous research runner.

The active agent uses Google ADK with the native Gemini model adapter.  The
legacy Builder/Strategist comparison runner still owns ``harness.provider``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_ADK_MODEL = "gemini-3.5-flash-lite"


@dataclass(frozen=True)
class ADKSettings:
    model: str
    provider: str = "google-adk"


def configure_google_adk_environment() -> ADKSettings:
    """Validate Gemini credentials and expose them under ADK's native name.

    ``GOOGLE_API_KEY`` / ``ADK_MODEL`` are canonical.  The old Gemini block's
    ``LLM_API_KEY`` / ``LLM_MODEL`` values remain accepted so existing local
    configuration continues to work during the migration.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        legacy_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        legacy_key = os.environ.get("LLM_API_KEY", "").strip()
        if legacy_key and legacy_provider not in ("", "gemini"):
            raise EnvironmentError(
                "The Google ADK runner cannot reuse LLM_API_KEY while "
                f"LLM_PROVIDER={legacy_provider!r}. Set GOOGLE_API_KEY to a Gemini API key."
            )
        api_key = legacy_key
    if not api_key:
        raise EnvironmentError(
            "Google ADK requires GOOGLE_API_KEY. Add a Gemini API key to .env "
            "(the legacy Gemini LLM_API_KEY name is also accepted)."
        )

    # Google ADK's native Gemini adapter reads this variable lazily.
    os.environ["GOOGLE_API_KEY"] = api_key
    model = (
        os.environ.get("ADK_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
        or DEFAULT_ADK_MODEL
    )
    return ADKSettings(model=model)
