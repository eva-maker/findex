import logging
from pathlib import Path

import pytest

from findex.models import DocMeta, Index, Posting
from findex.ranking import matching_ids, search
from findex.timing import timed


@pytest.fixture
def index() -> Index:
    postings = {
        "common": [Posting(d, 1) for d in range(3)],
        "rare": [Posting(2, 1)],
    }
    doc_meta = {d: DocMeta(d, Path(f"doc{d}.txt"), 100) for d in range(3)}
    return Index(postings, doc_meta)


def test_timed_keeps_name_and_doc():
    @timed
    def search_like():
        """Find."""

    assert search_like.__name__ == "search_like"
    assert search_like.__doc__ == "Find."


def test_timed_logs_even_on_exception(caplog):
    @timed
    def boom():
        raise ValueError

    with caplog.at_level(logging.DEBUG, logger="findex"):
        with pytest.raises(ValueError):
            boom()
    assert "boom took" in caplog.text


def test_index_is_hashable_by_identity(index):
    assert hash(index) == hash(index)
    assert {index: 1}[index] == 1


def test_repeated_query_is_cache_hit(index):
    matching_ids.cache_clear()
    search(index, "common OR rare")
    hits = matching_ids.cache_info().hits
    search(index, "common OR rare")
    assert matching_ids.cache_info().hits == hits + 1


def test_cache_hit_is_logged(index, caplog):
    matching_ids.cache_clear()
    with caplog.at_level(logging.DEBUG, logger="findex"):
        search(index, "common")
        search(index, "common")
    assert "miss" in caplog.text and "HIT" in caplog.text


def test_cached_values_are_immutable(index):
    terms, ids = matching_ids(index, "common")
    assert isinstance(terms, tuple) and isinstance(ids, frozenset)