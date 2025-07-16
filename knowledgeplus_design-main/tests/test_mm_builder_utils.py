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


def test_load_model_and_processor_caches(monkeypatch):
    calls = {"model": 0, "processor": 0}

    class DummyModel:
        pass

    class DummyProcessor:
        pass

    def fake_model(name):
        calls["model"] += 1
        return DummyModel()

    def fake_processor(name):
        calls["processor"] += 1
        return DummyProcessor()

    dummy_transformers = types.SimpleNamespace(
        CLIPModel=types.SimpleNamespace(from_pretrained=fake_model),
        CLIPProcessor=types.SimpleNamespace(from_pretrained=fake_processor),
    )
    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)

    mm_builder_utils._clip_model = None
    mm_builder_utils._clip_processor = None

    m1, p1 = mm_builder_utils.load_model_and_processor()
    m2, p2 = mm_builder_utils.load_model_and_processor()

    assert isinstance(m1, DummyModel)
    assert isinstance(p1, DummyProcessor)
    assert m1 is m2
    assert p1 is p2
    assert calls["model"] == 1
    assert calls["processor"] == 1


def test_get_image_embedding(monkeypatch):
    expected = [0.1] * mm_builder_utils.config.EMBEDDING_DIM

    class DummyFeatures:
        def __init__(self, vec):
            self.vec = vec

        def detach(self):
            return self

        def cpu(self):
            return self

        def __getitem__(self, idx):
            return DummyList(self.vec)


    class DummyList(list):
        def tolist(self):
            return list(self)

    class DummyModel:
        def get_image_features(self, **kwargs):
            return DummyFeatures(expected)

    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"dummy": True}

    monkeypatch.setattr(
        mm_builder_utils,
        "load_model_and_processor",
        lambda: (DummyModel(), DummyProcessor()),
    )

    result = mm_builder_utils.get_image_embedding(object())
    assert result == expected
