import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex_lab_01.corpus import iter_documents
from findex_lab_01.tokens import tokenize


def compute_stats(corpus_dir: Path) -> dict:

    doc_count = 0
    token_count = 0
    term_counts = Counter()

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


def run_with_measurement(corpus_dir: Path) -> dict:
    tracemalloc.start()
    start = time.perf_counter()

    stats = compute_stats(corpus_dir)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats["elapsed_seconds"] = round(elapsed, 4)
    stats["peak_memory_mb"] = round(peak / 1024 / 1024, 2)
    return stats


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