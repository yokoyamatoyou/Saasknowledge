import types
from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import pytest

from shared.chat_controller import ChatController

class DummyClient:
    def __init__(self, text: str):
        self._text = text
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **k: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._text))]
                )
            )
        )


def test_generate_conversation_title_strips_quotes():
    controller = ChatController(None)  # type: ignore[arg-type]
    client = DummyClient("「Session Title」")
    title = controller.generate_conversation_title(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        client=client,
    )
    assert title == "Session Title"


def test_generate_gpt_response_raises_without_client(monkeypatch):
    controller = ChatController(None)  # type: ignore[arg-type]

    # Force internal client initialization failure
    monkeypatch.setattr(
        ChatController,
        "_get_openai_client_internal",
        staticmethod(lambda: None),
    )

    gen = controller.generate_gpt_response("hello", conversation_history=[])
    with pytest.raises(RuntimeError):
        next(gen)
