import numpy as np
import pytest

from findex.models import Index
from findex.ranking import ScorerName, make_scorer, search
from findex.vectorized import search_vectorized, top_k, vectors_for

QUERIES = [
    "aaa",
    "aab OR aac",
    "aaa AND aab",
    "aaa AND NOT aab",
    "aad OR aae OR aaf",
    "aaz",
    "zzzz",
]


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("query", QUERIES)
def test_same_ranking_as_lab3(synthetic_index: Index, query: str, scorer: ScorerName) -> None:
    k = synthetic_index.num_docs
    old = search(synthetic_index, query, scorer=make_scorer(scorer), k=k)
    new = search_vectorized(synthetic_index, query, scorer, k)
    old_scores = {r.doc_id: r.score for r in old}
    new_scores = {r.doc_id: r.score for r in new}
    assert old_scores.keys() == new_scores.keys()
    for doc_id, score in old_scores.items():
        assert new_scores[doc_id] == pytest.approx(score, rel=1e-9)
    assert [r.doc_id for r in new] == sorted(new_scores, key=lambda d: (-new_scores[d], d))
    assert [r.title for r in new] == [synthetic_index.doc(r.doc_id).path.stem for r in new]


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_top_k_is_prefix_of_full_ranking(synthetic_index: Index, scorer: ScorerName) -> None:
    query = "aaa OR aab OR aac"
    full = search_vectorized(synthetic_index, query, scorer, synthetic_index.num_docs)
    assert search_vectorized(synthetic_index, query, scorer, 5) == full[:5]


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_tiny_index_matches_lab3(tiny_index: Index, scorer: ScorerName) -> None:
    for query in ("cat", "cat OR dog", "the", "old man", "cat AND NOT dog"):
        old = search(tiny_index, query, scorer=make_scorer(scorer), k=10)
        new = search_vectorized(tiny_index, query, scorer, 10)
        assert {r.doc_id for r in old} == {r.doc_id for r in new}
        old_scores = {r.doc_id: r.score for r in old}
        for result in new:
            assert result.score == pytest.approx(old_scores[result.doc_id])


def test_no_results_for_unknown_term(tiny_index: Index) -> None:
    assert search_vectorized(tiny_index, "unicorn") == []


def test_top_k_breaks_ties_by_doc_id() -> None:
    candidates = np.array([5, 3, 9, 1], dtype=np.int64)
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    assert top_k(candidates, scores, 2).tolist() == [1, 3]
    assert top_k(candidates, np.array([1.0, 2.0, 0.5, 2.0]), 3).tolist() == [1, 3, 5]


def test_vector_index_layout(tiny_index: Index) -> None:
    vix = vectors_for(tiny_index)
    assert vix.doc_ids.dtype == np.int32 and vix.tfs.dtype == np.int32
    assert vix.doc_len.dtype == np.int32
    found = vix.postings("cat")
    assert found is not None
    ids, tfs = found
    assert ids.tolist() == [p.doc_id for p in tiny_index["cat"]]
    assert tfs.tolist() == [p.freq for p in tiny_index["cat"]]
    assert vix.postings("unicorn") is None
    assert vectors_for(tiny_index) is vix
