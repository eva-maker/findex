"""Lab 8, M2: Lab 3 scorer vs NumPy scorer, and a search page with snippets before/after.

Run:
    uv run python -m findex.numpy_benchmark index.bin --corpus data/raw --synthetic-docs 50000

Timings are the median of --repeat calls with a warm Boolean cache, so only the scoring and
top-k parts are compared. Prints markdown tables for the README.
"""

import argparse
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from findex.models import DocMeta, Index, Posting
from findex.ranking import matching_ids, search
from findex.serialize import load_index
from findex.snippets import load_doc_tokens
from findex.vectorized import search_vectorized, vectors_for
from findex.web.service import run_search, run_search_baseline


def median_ms(fn: Callable[[], object], repeat: int) -> float:
    times: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def scorer_row(index: Index, label: str, query: str, repeat: int) -> str:
    _terms, matched = matching_ids(index, query)
    vectors_for(index)
    search_vectorized(index, query)
    old = median_ms(lambda: search(index, query), repeat)
    new = median_ms(lambda: search_vectorized(index, query), repeat)
    return f"| {label} | {len(matched):_} | {old:.3f} | {new:.3f} | {old / new:.1f}x |"


def real_queries(index: Index) -> list[tuple[str, str]]:
    by_df = sorted(index, key=index.df)
    common = by_df[-1]
    rare = next(t for t in by_df if index.df(t) == 1) if index.df(by_df[0]) == 1 else by_df[0]
    middle = [t for t in by_df if 1 < index.df(t) < index.num_docs]
    triple = middle[len(middle) // 2 : len(middle) // 2 + 3]
    return [
        (f"common term `{common}`", common),
        (f"rare term `{rare}`", rare),
        (f"3 terms `{' '.join(triple)}`", " OR ".join(triple)),
    ]


def synthetic_index(num_docs: int, seed: int = 1) -> tuple[Index, list[tuple[str, str]]]:
    rng = random.Random(seed)
    meta = {i: DocMeta(i, Path(f"d{i}.txt"), rng.randint(5_000, 200_000)) for i in range(num_docs)}
    shares = {"common": 0.9, "alpha": 0.3, "bravo": 0.3, "charlie": 0.3, "rare": 5 / num_docs}
    postings: dict[str, list[Posting]] = {}
    for term, share in shares.items():
        ids = sorted(rng.sample(range(num_docs), max(1, int(num_docs * share))))
        postings[term] = [Posting(i, rng.randint(1, 30)) for i in ids]
    queries = [
        ("common term", "common"),
        ("rare term", "rare"),
        ("3 terms", "alpha OR bravo OR charlie"),
    ]
    return Index(postings, meta), queries


def page_rows(index: Index, corpus: Path, query: str, repeat: int) -> list[str]:
    run_search_baseline(index, corpus, query, 10, 1, "bm25")
    baseline = median_ms(lambda: run_search_baseline(index, corpus, query, 10, 1, "bm25"), repeat)
    load_doc_tokens.cache_clear()
    started = time.perf_counter()
    run_search(index, corpus, query, 10, 1, "bm25")
    cold = (time.perf_counter() - started) * 1000
    warm = median_ms(lambda: run_search(index, corpus, query, 10, 1, "bm25"), repeat)
    return [
        f"| Lab 7 (tokenize book on every request) | {baseline:.1f} | 1.0x |",
        f"| Lab 8, first request (builds token arrays) | {cold:.1f} | {baseline / cold:.1f}x |",
        f"| Lab 8, warm | {warm:.1f} | {baseline / warm:.1f}x |",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--synthetic-docs", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    header = "| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |\n|---|---|---|---|---|"
    index = load_index(args.index)
    print(f"Real index: {index}\n")
    print(header)
    for label, query in real_queries(index):
        print(scorer_row(index, label, query, args.repeat))

    if args.synthetic_docs:
        big, queries = synthetic_index(args.synthetic_docs)
        print(f"\nSynthetic index with {args.synthetic_docs:_} documents\n")
        print(header)
        for label, query in queries:
            print(scorer_row(big, label, query, max(20, args.repeat // 10)))

    if args.corpus is not None:
        query = real_queries(index)[2][1]
        print(f"\nSearch page (k=10, with snippets), query `{query}`\n")
        print("| Version | ms per page | Speedup |\n|---|---|---|")
        print("\n".join(page_rows(index, args.corpus, query, max(5, args.repeat // 20))))


main()
