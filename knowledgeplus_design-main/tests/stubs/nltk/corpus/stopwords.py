"""Very small subset of English stopwords for tests."""

_BASIC_STOPWORDS = {"this", "the", "a", "is"}


def words(lang: str | None = None) -> list[str]:
    """Return a small list of stopwords.

    Parameters mirror the real NLTK API but only ``english`` is supported.
    """

    if lang is not None and lang != "english":
        return []
    return list(_BASIC_STOPWORDS)
