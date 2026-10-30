import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import TypedDict

from findex.corpus import iter_documents
from findex.models import Index
from findex.tokens import tokenize


class CorpusStats(TypedDict):
    """Форма словника зі статистикою: pyright знає, які ключі й якого типу."""

    doc_count: int
    token_count: int
    vocab_size: int
    top_50: list[tuple[str, int]]


class MeasuredStats(CorpusStats):
    elapsed_seconds: float
    peak_memory_mb: float


class IndexStats(TypedDict):
    doc_count: int
    token_count: int
    vocab_size: int
    avg_doc_length: float
    has_positions: bool
    top_terms: list[tuple[str, int]]


def index_stats(index: Index, top: int = 10) -> IndexStats:
    """Числа Лаби 1, але з готового індексу (без повторного читання книжок)."""
    totals = Counter({term: sum(p.freq for p in plist) for term, plist in index.items()})
    return {
        "doc_count": index.num_docs,
        "token_count": sum(meta.length for meta in index.docs()),
        "vocab_size": len(index),
        "avg_doc_length": index.avg_doc_length,
        "has_positions": index.has_positions,
        "top_terms": totals.most_common(top),
    }


def compute_stats(corpus_dir: Path) -> CorpusStats:
    doc_count = 0
    token_count = 0
    term_counts: Counter[str] = Counter()

    for doc in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(doc):
            token_count += 1
            term_counts[token] += 1

    return {
        "doc_count": doc_count,
        "token_count": token_count,
        "vocab_size": len(term_counts),
        "top_50": term_counts.most_common(50),
    }


def run_with_measurement(corpus_dir: Path) -> MeasuredStats:
    tracemalloc.start()
    start = time.perf_counter()

    stats = compute_stats(corpus_dir)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        **stats,
        "elapsed_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak / 1024 / 1024, 2),
    }


if __name__ == "__main__":
    stats = run_with_measurement(Path("data/raw"))
    print(f"Documents:   {stats['doc_count']}")
    print(f"Tokens:      {stats['token_count']}")
    print(f"Vocab size:  {stats['vocab_size']}")
    print(f"Time:        {stats['elapsed_seconds']}s")
    print(f"Peak memory: {stats['peak_memory_mb']} MB")
    print("\nTop 50 terms:")
    for term, count in stats["top_50"]:
        print(f"  {term}: {count}")
