import json
import sys
import types
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import suggest_synonyms  # noqa: E402


class DummyClient:
    def __init__(self):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    def create(self, *, messages, **_):
        term = messages[-1]["content"].split("単語: ")[-1]
        data = [f"{term}_1", f"{term}_2"]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(data)))]
        )


def test_suggest_synonyms_returns_expected(tmp_path, monkeypatch):
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"existing": ["x"]}, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(suggest_synonyms, "load_zero_hit_queries", lambda p: ["foo", "existing", "bar"])
    monkeypatch.setattr(suggest_synonyms, "get_openai_client", lambda: DummyClient())

    result = suggest_synonyms.suggest_synonyms(Path("zero.log"), syn)
    assert result == {
        "foo": ["foo_1", "foo_2"],
        "bar": ["bar_1", "bar_2"],
    }


def test_cli_update_merges_terms(tmp_path, monkeypatch):
    syn = tmp_path / "syn.json"
    syn.write_text(json.dumps({"existing": ["v"]}, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(suggest_synonyms, "load_zero_hit_queries", lambda p: ["new"])
    monkeypatch.setattr(suggest_synonyms, "get_openai_client", lambda: DummyClient())

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--log",
        str(tmp_path / "zero.log"),
        "--thesaurus",
        str(syn),
        "--update",
    ])

    suggest_synonyms.main()

    data = json.loads(syn.read_text(encoding="utf-8"))
    assert data == {
        "existing": ["v"],
        "new": ["new_1", "new_2"],
    }
