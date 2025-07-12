"""Utility functions for initializing OpenAI clients."""

import logging
from typing import List, Optional

from config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

from .upload_utils import ensure_openai_key

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai may be missing in tests
    OpenAI = None


def _create_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        logger.error("OpenAI package is not available")
        return None
    try:
        api_key = ensure_openai_key()
    except Exception as e:
        logger.error("Failed to load OpenAI API key: %s", e, exc_info=True)
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error("Failed to initialize OpenAI client: %s", e, exc_info=True)
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


def get_embeddings_batch(
    texts: List[str],
    client,
    model: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> List[List[float]]:
    """Return embeddings for ``texts`` using a single API call."""
    if client is None or not texts:
        return []
    try:
        response = client.embeddings.create(
            model=model,
            input=texts,
            dimensions=dimensions,
        )
        return [d.embedding for d in response.data]
    except Exception as e:  # pragma: no cover - network issues
        logger.error("Batch embedding error: %s", e)
        return []
