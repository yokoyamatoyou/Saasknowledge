import types
from pathlib import Path
import sys
import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.prompt_advisor import generate_prompt_advice

class MockClient:
    def __init__(self, text):
        self._text = text
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **k: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._text))]
                )
            )
        )

def test_generate_prompt_advice_returns_text():
    client = MockClient("- improved")
    result = generate_prompt_advice("hello", client=client)
    assert result == "- improved"


def test_generate_prompt_advice_uses_openai(monkeypatch):
    """generate_prompt_advice should call OpenAI when no client is supplied."""
    import openai
    from shared import prompt_advisor

    advice = "use more detail"

    class DummyOpenAI:
        def __init__(self, api_key=None):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **_: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            message=types.SimpleNamespace(content=advice)
                        )]
                    )
                )
            )

    monkeypatch.setattr(prompt_advisor, "OpenAI", DummyOpenAI)
    monkeypatch.setattr(prompt_advisor, "ensure_openai_key", lambda: "key")

    result = prompt_advisor.generate_prompt_advice("prompt")
    assert isinstance(result, str) and result
