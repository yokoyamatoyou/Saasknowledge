import os
import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_env_reads_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    import shared.env as env
    importlib.reload(env)
    env.load_env(env_file)
    assert os.getenv("FOO") == "bar"
