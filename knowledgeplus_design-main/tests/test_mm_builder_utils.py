import types
from pathlib import Path
import sys

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from core import mm_builder_utils


def test_analyze_image_with_gpt4o_parses_json(monkeypatch):
    # Prepare fake OpenAI client returning JSON text
    def fake_create(**kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"foo": "bar"}'))]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )

    monkeypatch.setattr(mm_builder_utils, "get_openai_client", lambda: client)

    result = mm_builder_utils.analyze_image_with_gpt4o("imgdata", "img.png")
    assert result == {"foo": "bar"}


def test_analyze_image_with_gpt4o_missing_client(monkeypatch):
    monkeypatch.setattr(mm_builder_utils, "get_openai_client", lambda: None)
    result = mm_builder_utils.analyze_image_with_gpt4o("b64", "file.png")
    assert result == {"error": "OpenAIクライアントが利用できません"}
