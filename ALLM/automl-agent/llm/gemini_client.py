"""
Gemini API client — handles communication with the Google Generative AI API.

Design decisions:
- API key read from environment variable or Streamlit secrets.
- Graceful fallback: if the API call fails, returns a user-friendly error
  message instead of crashing the app.
- Temperature set to 0.3 for factual, grounded responses.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.0-flash"


def _get_api_key() -> str | None:
    """Resolve the Gemini API key from environment or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def ask_gemini(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Send a prompt to Gemini and return the response text.

    Returns a fallback message on any failure so the UI never crashes.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "⚠️ Gemini API key not configured. "
            "Set `GEMINI_API_KEY` in your `.env` file or Streamlit secrets."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            _GEMINI_MODEL,
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text

    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        return f"⚠️ Gemini request failed: {exc}. Check your API key and network connection."
