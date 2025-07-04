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
