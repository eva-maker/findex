import math
from pathlib import Path

import pytest

from findex.models import DocMeta, Index, Posting
from findex.ranking import BM25, SearchResult, TfIdf, search


@pytest.fixture
def index() -> Index:
    # 5 документів. "common" є в усіх, "rare" лише в документі 2,
    # "x" є в документах 0 (коротка) і 1 (дуже довга).
    postings = {
        "common": [Posting(d, 1) for d in range(5)],
        "rare": [Posting(2, 1)],
        "x": [Posting(0, 1), Posting(1, 1)],
    }
    lengths = {0: 10, 1: 1000, 2: 100, 3: 100, 4: 100}
    doc_meta = {d: DocMeta(d, Path(f"doc{d}.txt"), n) for d, n in lengths.items()}
    return Index(postings, doc_meta)


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_outweighs_common_term(index, scorer):
    rare = scorer("rare", Posting(2, 1), index)
    common = scorer("common", Posting(2, 1), index)
    assert rare > common


def test_doc_with_rare_term_ranks_first(index):
    results = search(index, "common rare")
    assert results[0].doc_id == 2


def test_bm25_saturation_20th_repeat_adds_almost_nothing(index):
    bm25 = BM25()
    s = lambda tf: bm25("x", Posting(2, tf), index)
    first_gain = s(1)
    gain_20th = s(20) - s(19)
    assert gain_20th < 0.02 * first_gain


def test_bm25_has_a_ceiling(index):
    bm25 = BM25()
    idf = math.log(1 + (5 - 2 + 0.5) / (2 + 0.5))      # N=5, df("x")=2
    ceiling = idf * (bm25.k1 + 1)
    huge = bm25("x", Posting(2, 10**9), index)
    assert huge < ceiling
    assert huge == pytest.approx(ceiling, rel=1e-6)    # прагне до стелі, але не перевищує


def test_bm25_short_document_beats_long_one(index):
    bm25 = BM25()
    short = bm25("x", Posting(0, 1), index)
    long_ = bm25("x", Posting(1, 1), index)
    assert short > long_


def test_bm25_with_b_zero_ignores_length(index):
    bm25 = BM25(b=0)
    short = bm25("x", Posting(0, 1), index)
    long_ = bm25("x", Posting(1, 1), index)
    assert short == pytest.approx(long_)


def test_any_callable_is_a_scorer(index):
    results = search(index, "rare", scorer=lambda term, posting, ix: 1.0)
    assert results[0].score == 1.0


def test_search_is_sorted_descending_and_respects_k(index):
    results = search(index, "common OR rare OR x", k=3)
    assert len(results) == 3
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_query_is_normalized_like_documents(index):
    assert search(index, "RARE")[0].doc_id == 2


def test_unknown_term_gives_empty_result(index):
    assert search(index, "python") == []


def test_search_result_ordering():
    low = SearchResult(1.0, 5, "a")
    high = SearchResult(3.0, 2, "b")
    assert sorted([high, low]) == [low, high]
    assert max(low, high) is high