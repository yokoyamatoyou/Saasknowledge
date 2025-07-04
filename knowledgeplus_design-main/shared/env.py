"""Environment utilities for loading configuration."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_env_loaded = False


def load_env(path: str | Path | None = None, *, force: bool = False) -> None:
    """Load environment variables from a .env file.

    Parameters
    ----------
    path
        Optional path to the ``.env`` file.  Defaults to ``.env`` in the
        repository root.
    force
        Reload even if variables were previously loaded.
    """

    global _env_loaded
    if _env_loaded and not force:
        return

    env_path = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    loaded = False
    if env_path.exists():
        loaded = bool(load_dotenv(env_path))
    _env_loaded = loaded
