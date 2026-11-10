import pytest

from findex.models import Index
from findex.ranking import ScorerName
from findex.web.service import run_search, run_search_baseline


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("query", ["aaa", "aab OR aac", "aad AND aae", "aaa AND NOT aab"])
@pytest.mark.parametrize("page", [1, 2])
def test_run_search_same_as_baseline(
    synthetic_index: Index, query: str, scorer: ScorerName, page: int
) -> None:
    fast = run_search(synthetic_index, None, query, 5, page, scorer)
    slow = run_search_baseline(synthetic_index, None, query, 5, page, scorer)
    assert fast.total == slow.total
    assert [h.doc_id for h in fast.hits] == [h.doc_id for h in slow.hits]
    assert [h.snippet for h in fast.hits] == [h.snippet for h in slow.hits]
    for a, b in zip(fast.hits, slow.hits, strict=True):
        assert a.score == pytest.approx(b.score, rel=1e-9)
