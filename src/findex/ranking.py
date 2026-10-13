import argparse
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Protocol

from findex.models import Index, Posting
from findex.query import QueryError, parse
from findex.serialize import open_index
from findex.tokens import tokenize


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


def search(
    index: Index,
    query: str,
    scorer: Scorer = BM25(),
    k: int = 10,
) -> list[SearchResult]:
    """Спершу query language визначає, які документи ПІДХОДЯТЬ (Boolean),
    потім scorer впорядковує їх за релевантністю."""
    tree = parse(query)
    matched = tree.evaluate(index)

    scores: defaultdict[int, float] = defaultdict(float)
    for term in set(tree.positive_terms()):          # терміни під NOT балів не дають
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
    args = parser.parse_args()

    scorer = BM25() if args.scorer == "bm25" else TfIdf()
    try:
        with open_index(args.index_path) as index:
            results = search(index, args.query, scorer=scorer, k=args.k)
    except QueryError as error:
        print(f"Помилка запиту: {error}")
        return

    print(f"Query: {args.query!r}  scorer={scorer}")
    if not results:
        print("Nothing found.")
    for rank, result in enumerate(results, start=1):
        print(f"{rank:2}. {result}")


if __name__ == "__main__":
    main()