import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(PROJECT_ROOT))


def test_load_env_reads_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    import shared.env as env

    importlib.reload(env)
    env.load_env(env_file)
    assert os.getenv("FOO") == "bar"


def test_load_env_then_reload(tmp_path, monkeypatch):
    """Calling load_env twice should load variables only when a file exists."""
    env_file = tmp_path / ".env"
    env_file.write_text("BAR=baz\n", encoding="utf-8")

    monkeypatch.delenv("BAR", raising=False)
    import shared.env as env

    importlib.reload(env)

    # First call without a file present does nothing
    env.load_env(Path("nonexistent"))
    assert os.getenv("BAR") is None

    # Second call with a valid file should load the variable
    env.load_env(env_file)
    assert os.getenv("BAR") == "baz"
