"""Utility functions for initializing OpenAI clients."""

from typing import Optional

from .upload_utils import ensure_openai_key

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai may be missing in tests
    OpenAI = None


def _create_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    try:
        api_key = ensure_openai_key()
    except Exception:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


try:
    import streamlit as st

    @st.cache_resource  # pragma: no cover - caching only used when Streamlit runs
    def get_openai_client() -> Optional["OpenAI"]:
        """Return a cached OpenAI client or ``None`` if initialization fails."""
        return _create_client()
except Exception:  # pragma: no cover - Streamlit not available

    def get_openai_client() -> Optional["OpenAI"]:
        """Return an OpenAI client or ``None`` if initialization fails."""
        return _create_client()
