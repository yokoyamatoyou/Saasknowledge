import json
import types

from shared import rule_extractor


class DummyClient:
    def __init__(self, payload):
        self.payload = payload
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=json.dumps(self.payload))
                )
            ]
        )


def test_extract_rules_basic():
    client = DummyClient([{"rule_id": "r1"}])
    rules = rule_extractor.extract_rules("text", client=client)
    assert rules == [{"rule_id": "r1"}]


def test_extract_rules_no_client(monkeypatch):
    monkeypatch.setattr(rule_extractor, "get_openai_client", lambda: None)
    assert rule_extractor.extract_rules("text", client=None) == []
