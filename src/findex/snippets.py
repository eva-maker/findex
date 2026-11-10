import functools
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from findex.corpus import read_document
from findex.tokens import normalize_text, token_spans, token_table

type Match = tuple[str, int, int]


def _find_matches(text: str, terms: Iterable[str]) -> list[Match]:
    wanted = set(terms)
    return [(t, s, e) for t, s, e in token_spans(text) if t in wanted]


def _best_center(matches: list[Match], width: int) -> int:
    best_start, best_score = matches[0][1], -1
    counts: dict[str, int] = {}
    end = 0
    for term, start, _e in matches:
        while end < len(matches) and matches[end][1] <= start + width:
            counts[matches[end][0]] = counts.get(matches[end][0], 0) + 1
            end += 1
        if len(counts) > best_score:
            best_start, best_score = start, len(counts)
        counts[term] -= 1
        if counts[term] == 0:
            del counts[term]
    return best_start


def _assemble(norm: str, matches: list[Match], width: int, mark: str) -> str:
    center = _best_center(matches, width)
    lo = max(0, center - width)
    hi = min(len(norm), center + width)
    while 0 < lo < center and not norm[lo - 1].isspace():
        lo += 1
    while center < hi < len(norm) and not norm[hi].isspace():
        hi -= 1
    pieces: list[str] = []
    pos = lo
    for _t, s, e in matches:
        if s >= lo and e <= hi:
            pieces += [norm[pos:s], mark, norm[s:e], mark]
            pos = e
    pieces.append(norm[pos:hi])
    body = " ".join("".join(pieces).split())
    return ("… " if lo > 0 else "") + body + (" …" if hi < len(norm) else "")


def make_snippet(text: str, terms: Iterable[str], width: int = 80, mark: str = "**") -> str:
    norm = normalize_text(text)
    matches = _find_matches(norm, terms)
    if not matches:
        return ""
    return _assemble(norm, matches, width, mark)


@dataclass(frozen=True, slots=True)
class DocTokens:
    text: str
    ids: NDArray[np.int32]
    starts: NDArray[np.int32]
    ends: NDArray[np.int32]
    vocab: dict[str, int]


def build_doc_tokens(raw_text: str) -> DocTokens:
    text = normalize_text(raw_text)
    tokens, spans = token_table(text)
    vocab: dict[str, int] = {}
    ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in tokens), np.int32, len(tokens))
    return DocTokens(text, ids, spans[:, 0].copy(), spans[:, 1].copy(), vocab)


@functools.lru_cache(maxsize=512)
def load_doc_tokens(path: Path) -> DocTokens:
    return build_doc_tokens(read_document(path))


def warm_doc_tokens(paths: Iterable[Path]) -> int:
    loaded = 0
    for path in paths:
        try:
            load_doc_tokens(path)
        except OSError:
            continue
        loaded += 1
    return loaded


def snippet_from_tokens(
    doc: DocTokens, terms: Iterable[str], width: int = 80, mark: str = "**"
) -> str:
    wanted = {t: doc.vocab[t] for t in set(terms) if t in doc.vocab}
    if not wanted:
        return ""
    hit = np.flatnonzero(np.isin(doc.ids, np.fromiter(wanted.values(), np.int32, len(wanted))))
    if hit.size == 0:
        return ""
    term_of = {i: t for t, i in wanted.items()}
    matches: list[Match] = list(
        zip(
            (term_of[i] for i in doc.ids[hit].tolist()),
            doc.starts[hit].tolist(),
            doc.ends[hit].tolist(),
            strict=True,
        )
    )
    return _assemble(doc.text, matches, width, mark)
