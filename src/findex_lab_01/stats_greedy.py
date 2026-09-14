import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex_lab_01.corpus import iter_documents
from findex_lab_01.tokens import tokenize


def compute_stats_greedy(corpus_dir: Path) -> dict:

    documents = list(iter_documents(corpus_dir))       # whole corpus held in memory at once
    all_tokens = []
    for doc in documents:
        all_tokens.extend(list(tokenize(doc)))          # whole token stream held in memory at once

    term_counts = Counter(all_tokens)

    return {
        "doc_count": len(documents),
        "token_count": len(all_tokens),
        "vocab_size": len(term_counts),
        "top_50": term_counts.most_common(50),
    }


def run_with_measurement_greedy(corpus_dir: Path) -> dict:
    tracemalloc.start()
    start = time.perf_counter()

    stats = compute_stats_greedy(corpus_dir)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats["elapsed_seconds"] = round(elapsed, 4)
    stats["peak_memory_mb"] = round(peak / 1024 / 1024, 2)
    return stats


if __name__ == "__main__":
    stats = run_with_measurement_greedy(Path("data/raw"))
    print(f"Documents:   {stats['doc_count']}")
    print(f"Tokens:      {stats['token_count']}")
    print(f"Vocab size:  {stats['vocab_size']}")
    print(f"Time:        {stats['elapsed_seconds']}s")
    print(f"Peak memory: {stats['peak_memory_mb']} MB")