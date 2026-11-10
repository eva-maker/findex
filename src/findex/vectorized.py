import math
import weakref
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from findex.models import Index
from findex.ranking import ScorerName, SearchResult, matching_ids


@dataclass(frozen=True, slots=True)
class VectorIndex:
    term_row: dict[str, int]
    offsets: NDArray[np.int64]
    doc_ids: NDArray[np.int32]
    tfs: NDArray[np.int32]
    doc_len: NDArray[np.int32]
    num_docs: int
    avg_len: float

    @classmethod
    def from_index(cls, index: Index) -> VectorIndex:
        terms = list(index)
        total = sum(len(index[t]) for t in terms)
        doc_ids = np.fromiter((p.doc_id for t in terms for p in index[t]), np.int32, total)
        tfs = np.fromiter((p.freq for t in terms for p in index[t]), np.int32, total)
        sizes = np.fromiter((len(index[t]) for t in terms), np.int64, len(terms))
        offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(sizes)))
        metas = list(index.docs())
        doc_len = np.zeros(max((m.doc_id for m in metas), default=-1) + 1, np.int32)
        for meta in metas:
            doc_len[meta.doc_id] = meta.length
        return cls(
            term_row={t: i for i, t in enumerate(terms)},
            offsets=offsets,
            doc_ids=doc_ids,
            tfs=tfs,
            doc_len=doc_len,
            num_docs=index.num_docs,
            avg_len=index.avg_doc_length,
        )

    def postings(self, term: str) -> tuple[NDArray[np.int32], NDArray[np.int32]] | None:
        row = self.term_row.get(term)
        if row is None:
            return None
        lo, hi = int(self.offsets[row]), int(self.offsets[row + 1])
        return self.doc_ids[lo:hi], self.tfs[lo:hi]

    def score(
        self,
        terms: list[str],
        scorer: ScorerName = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        scores = np.zeros(len(self.doc_len), np.float64)
        touched = np.zeros(len(self.doc_len), np.bool_)
        for term in dict.fromkeys(terms):
            found = self.postings(term)
            if found is None:
                continue
            ids, tf = found
            df = ids.size
            if scorer == "bm25":
                idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
                norm = k1 * (1 - b + b * self.doc_len[ids] / self.avg_len)
                scores[ids] += idf * tf * (k1 + 1) / (tf + norm)
            else:
                scores[ids] += (1 + np.log(tf)) * math.log(self.num_docs / df)
            touched[ids] = True
        return scores, touched


_CACHE: weakref.WeakKeyDictionary[Index, VectorIndex] = weakref.WeakKeyDictionary()


def vectors_for(index: Index) -> VectorIndex:
    cached = _CACHE.get(index)
    if cached is None:
        cached = _CACHE[index] = VectorIndex.from_index(index)
    return cached


def top_k(candidates: NDArray[np.int64], scores: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    n = candidates.size
    if k < n:
        boundary = scores[np.argpartition(scores, n - k)[n - k]]
        keep = np.flatnonzero(scores >= boundary)
        candidates, scores = candidates[keep], scores[keep]
    order = np.lexsort((candidates, -scores))[:k]
    return candidates[order]


def search_vectorized(
    index: Index, query: str, scorer: ScorerName = "bm25", k: int = 10
) -> list[SearchResult]:
    terms, matched = matching_ids(index, query)
    vix = vectors_for(index)
    scores, touched = vix.score(list(terms), scorer)
    candidates = np.fromiter(matched, np.int64, len(matched))
    candidates = candidates[touched[candidates]]
    if candidates.size == 0:
        return []
    best = top_k(candidates, scores[candidates], k)
    return [
        SearchResult(float(scores[doc_id]), int(doc_id), index.doc(int(doc_id)).path.stem)
        for doc_id in best
    ]
