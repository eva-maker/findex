"""Lab 8: tokenize before/after on the whole corpus and memory cost of the snippet arrays.

Run: uv run python scripts/measure_lab8.py data/raw
"""

import statistics
import sys
import time
import tracemalloc
from collections.abc import Iterator
from pathlib import Path

from findex.corpus import list_document_paths, read_document
from findex.snippets import load_doc_tokens
from findex.tokens import normalize_text, token_spans, tokenize


def tokenize_lab7(text: str) -> Iterator[str]:
    for token, _start, _end in token_spans(normalize_text(text)):
        yield token


def median_seconds(texts: list[str], fn: object, repeat: int = 5) -> float:
    assert callable(fn)
    times: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        for text in texts:
            for _token in fn(text):
                pass
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def main() -> None:
    paths = list_document_paths(Path(sys.argv[1]))
    texts = [read_document(p) for p in paths]
    old = median_seconds(texts, tokenize_lab7)
    new = median_seconds(texts, tokenize)
    print(f"tokenize, {len(texts)} books: Lab 7 {old:.2f} s, Lab 8 {new:.2f} s, {old / new:.2f}x")

    load_doc_tokens.cache_clear()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    for path in paths:
        load_doc_tokens(path)
    elapsed = time.perf_counter() - started
    after, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tokens = sum(load_doc_tokens(p).ids.size for p in paths)
    print(f"snippet arrays: {tokens:_} tokens, built in {elapsed:.1f} s")
    print(
        f"memory kept: {(after - before) / 2**20:.1f} MiB, peak while building: {peak / 2**20:.1f} MiB"
    )


main()
