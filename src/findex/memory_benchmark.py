import gc
import tracemalloc
from array import array
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents_with_meta
from findex.tokens import tokenize


@dataclass  # без slots, без frozen — звичайний dataclass
class PostingPlain:
    doc_id: int
    freq: int


@dataclass(frozen=True, slots=True)
class PostingSlots:
    doc_id: int
    freq: int


def _build[P](corpus_dir: Path, make_posting: Callable[[int, int], P]) -> dict[str, list[P]]:
    """Спільний каркас для двох варіантів з об'єктами.

    [P] — generic за PEP 695: P це або PostingPlain, або PostingSlots, і pyright
    знає, що словник на виході містить саме той тип, який створює make_posting.
    """
    postings: defaultdict[str, list[P]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents_with_meta(corpus_dir)):
        term_freq: defaultdict[str, int] = defaultdict(int)
        for token in tokenize(doc.text):
            term_freq[token] += 1
        for term, freq in term_freq.items():
            postings[term].append(make_posting(doc_id, freq))
    return postings


def build_plain(corpus_dir: Path) -> dict[str, list[PostingPlain]]:
    return _build(corpus_dir, PostingPlain)


def build_slots(corpus_dir: Path) -> dict[str, list[PostingSlots]]:
    return _build(corpus_dir, PostingSlots)


def build_array(corpus_dir: Path) -> dict[str, array[int]]:
    """Кожен термін -> один плаский array('I') з парами doc_id, freq, doc_id, freq, ..."""
    postings: defaultdict[str, array[int]] = defaultdict(lambda: array("I"))
    for doc_id, doc in enumerate(iter_documents_with_meta(corpus_dir)):
        term_freq: defaultdict[str, int] = defaultdict(int)
        for token in tokenize(doc.text):
            term_freq[token] += 1
        for term, freq in term_freq.items():
            postings[term].append(doc_id)
            postings[term].append(freq)
    return postings


def measure(name: str, build_fn: Callable[[Path], object], corpus_dir: Path) -> None:
    gc.collect()
    tracemalloc.start()
    result = build_fn(corpus_dir)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{name:<20} peak: {peak / 1024 / 1024:>8.2f} MB")
    del result
    gc.collect()  # звільнити пам'ять перед наступним заміром


if __name__ == "__main__":
    corpus = Path("data/raw")
    measure("plain dataclass", build_plain, corpus)
    measure("slots dataclass", build_slots, corpus)
    measure("array('I')", build_array, corpus)
