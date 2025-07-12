import importlib
import sys
import types
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.chat_controller import ChatController  # noqa: E402


def test_title_generation_respects_env(monkeypatch):
    monkeypatch.setenv("TITLE_MODEL", "dummy-model")
    import config

    importlib.reload(config)

    captured = {}

    def create_mock(model=None, messages=None, temperature=None, max_tokens=None):
        captured["model"] = model
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="title"))
            ]
        )

    dummy_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create_mock)
        )
    )

    controller = ChatController(None)  # type: ignore[arg-type]
    controller.generate_conversation_title(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        client=dummy_client,
    )

    assert captured.get("model") == "dummy-model"
