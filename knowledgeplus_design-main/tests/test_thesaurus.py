import json

from shared.thesaurus import load_synonyms, update_synonyms


def test_load_synonyms_missing(tmp_path):
    path = tmp_path / "syn.json"
    assert load_synonyms(path) == {}


def test_update_synonyms_creates_and_loads(tmp_path):
    path = tmp_path / "syn.json"
    updated = update_synonyms("foo", ["bar", "baz"], path)
    assert updated == {"foo": ["bar", "baz"]}
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == updated
