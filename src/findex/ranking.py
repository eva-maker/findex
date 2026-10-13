import argparse
import functools
import heapq
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Protocol

from findex.corpus import read_document
from findex.models import Index, Posting
from findex.query import QueryError, parse
from findex.serialize import open_index
from findex.snippets import make_snippet
from findex.timing import log, timed


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True)
class TfIdf:
    """вага = (1 + log tf) * log(N / df)"""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        tf = 1 + math.log(posting.freq)
        return tf * idf


@dataclass(frozen=True)
class BM25:
    """TF-IDF з насиченням TF (k1) і штрафом за довжину документа (b)."""

    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        n = index.num_docs
        df = index.df(term)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

        doc_len = index.doc_length(posting.doc_id)
        length_norm = 1 - self.b + self.b * doc_len / index.avg_doc_length

        tf = posting.freq
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)


@dataclass(frozen=True, order=True)
class SearchResult:
    # order=True порівнює поля по черзі, зверху вниз, тому score — першим:
    # тоді sorted(results) / max(results) впорядковують за релевантністю.
    score: float
    doc_id: int
    title: str

    def __str__(self) -> str:
        return f"{self.score:7.3f}  [{self.doc_id}] {self.title}"


@functools.lru_cache(maxsize=256)
def matching_ids(index: Index, query: str) -> tuple[tuple[str, ...], frozenset[int]]:
    """Boolean-частина пошуку, закешована. Повертає незмінні tuple/frozenset:
    кеш віддає один і той самий об'єкт усім, тож його не можна було б мутувати."""
    tree = parse(query)
    return tuple(tree.positive_terms()), frozenset(tree.evaluate(index))


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer = BM25(),
    k: int = 10,
) -> list[SearchResult]:
    """Спершу query language визначає, які документи ПІДХОДЯТЬ (Boolean),
    потім scorer впорядковує їх за релевантністю."""
    hits_before = matching_ids.cache_info().hits
    terms, matched = matching_ids(index, query)
    cache_hit = matching_ids.cache_info().hits > hits_before
    log.debug("запит %r: кеш %s", query, "HIT" if cache_hit else "miss")

    scores: defaultdict[int, float] = defaultdict(float)
    for term in set(terms):                          # терміни під NOT балів не дають
        for posting in index.get(term, ()):
            if posting.doc_id in matched:
                scores[posting.doc_id] += scorer(term, posting, index)

    # топ-k без повного сортування: O(n log k)
    top = heapq.nlargest(k, scores.items(), key=itemgetter(1))
    return [
        SearchResult(score, doc_id, index.doc(doc_id).path.stem)
        for doc_id, score in top
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranked search over a saved index.")
    parser.add_argument("index_path", type=Path)
    parser.add_argument("query", type=str)
    parser.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true", help="показати лог часу й кешу")
    parser.add_argument("--repeat", type=int, default=1, help="повторити запит N разів (щоб побачити кеш)")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    scorer = BM25() if args.scorer == "bm25" else TfIdf()
    try:
        with open_index(args.index_path) as index:
            for _ in range(args.repeat):
                results = search(index, args.query, scorer=scorer, k=args.k)
            terms, _matched = matching_ids(index, args.query)

            print(f"Query: {args.query!r}  scorer={scorer}")
            if not results:
                print("Nothing found.")
            for rank, result in enumerate(results, start=1):
                print(f"{rank:2}. {result}")
                text = read_document(index.doc(result.doc_id).path)
                snippet = make_snippet(text, terms)
                if snippet:
                    print(f"      {snippet}")
    except QueryError as error:
        print(f"Помилка запиту: {error}")
        return


if __name__ == "__main__":
    main()