import sys
import types
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from core import mm_builder_utils  # noqa: E402


def test_analyze_image_with_gpt4o_parses_json(monkeypatch):
    # Prepare fake OpenAI client returning JSON text
    def fake_create(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"foo": "bar"}')
                )
            ]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    monkeypatch.setattr(mm_builder_utils, "get_openai_client", lambda: client)

    result = mm_builder_utils.analyze_image_with_gpt4o("imgdata", "img.png")
    assert result == {"foo": "bar"}


def test_analyze_image_with_gpt4o_missing_client(monkeypatch):
    monkeypatch.setattr(mm_builder_utils, "get_openai_client", lambda: None)
    result = mm_builder_utils.analyze_image_with_gpt4o("b64", "file.png")
    assert result == {"error": "OpenAIクライアントが利用できません"}


def test_get_embedding_handles_timeout(monkeypatch):
    """get_embedding should return None if the OpenAI call times out."""

    class DummyClient:
        def __init__(self):
            def raise_timeout(**_):
                raise TimeoutError("timeout")

            self.embeddings = types.SimpleNamespace(create=raise_timeout)

    monkeypatch.setattr(mm_builder_utils, "get_openai_client", lambda: DummyClient())

    result = mm_builder_utils.get_embedding("hello")
    assert result is None
