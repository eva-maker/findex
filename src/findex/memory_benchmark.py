import gc
import tracemalloc
from array import array
from collections import defaultdict
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


def _build(corpus_dir: Path, add_posting):
    postings = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents_with_meta(corpus_dir)):
        term_freq = defaultdict(int)
        for token in tokenize(doc.text):
            term_freq[token] += 1
        for term, freq in term_freq.items():
            add_posting(postings, term, doc_id, freq)
    return postings


def build_plain(corpus_dir: Path) -> dict:
    return _build(corpus_dir, lambda p, t, d, f: p[t].append(PostingPlain(d, f)))


def build_slots(corpus_dir: Path) -> dict:
    return _build(corpus_dir, lambda p, t, d, f: p[t].append(PostingSlots(d, f)))


def build_array(corpus_dir: Path) -> dict:
    """Кожен термін -> один плаский array('I') з парами doc_id, freq, doc_id, freq, ..."""
    postings = defaultdict(lambda: array("I"))
    for doc_id, doc in enumerate(iter_documents_with_meta(corpus_dir)):
        term_freq = defaultdict(int)
        for token in tokenize(doc.text):
            term_freq[token] += 1
        for term, freq in term_freq.items():
            postings[term].append(doc_id)
            postings[term].append(freq)
    return postings


def measure(name: str, build_fn, corpus_dir: Path):
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