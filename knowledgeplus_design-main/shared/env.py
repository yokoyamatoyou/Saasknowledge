from __future__ import annotations

"""Environment utilities for loading configuration."""

from pathlib import Path
from dotenv import load_dotenv

_env_loaded = False


def load_env(path: str | Path | None = None) -> None:
    """Load environment variables from a .env file once."""
    global _env_loaded
    if _env_loaded:
        return

    env_path = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _env_loaded = True
