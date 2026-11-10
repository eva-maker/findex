import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from markupsafe import Markup

from findex.corpus import read_document
from findex.models import Index
from findex.ranking import ScorerName, make_scorer, matching_ids, search
from findex.semantic import (
    Embedder,
    SemanticIndex,
    chunk_text,
    keyword_query,
    rrf,
    semantic_rank,
)
from findex.snippets import load_doc_tokens, make_snippet, snippet_from_tokens
from findex.vectorized import search_vectorized

_MARK = "\x00"

Mode = Literal["keyword", "semantic", "hybrid"]


@dataclass(frozen=True, slots=True)
class Hit:
    doc_id: int
    title: str
    score: float
    snippet: str  # з маркерами _MARK навколо збігів


@dataclass(frozen=True, slots=True)
class Page:
    total: int
    took_ms: float
    hits: list[Hit]


def document_path(index: Index, doc_id: int, corpus_dir: Path | None) -> Path:
    path = index.doc(doc_id).path
    return corpus_dir / path.name if corpus_dir is not None else path


def run_search(
    index: Index,
    corpus_dir: Path | None,
    query: str,
    k: int,
    page: int,
    scorer: ScorerName,
) -> Page:
    started = time.perf_counter()
    terms, matched = matching_ids(index, query)
    ranked = search_vectorized(index, query, scorer, k * page)[(page - 1) * k :]
    hits = [
        Hit(
            doc_id=result.doc_id,
            title=result.title,
            score=result.score,
            snippet=snippet_from_tokens(
                load_doc_tokens(document_path(index, result.doc_id, corpus_dir)),
                terms,
                mark=_MARK,
            ),
        )
        for result in ranked
    ]
    return Page(len(matched), (time.perf_counter() - started) * 1000, hits)


def run_search_baseline(
    index: Index,
    corpus_dir: Path | None,
    query: str,
    k: int,
    page: int,
    scorer: ScorerName,
) -> Page:
    started = time.perf_counter()
    terms, matched = matching_ids(index, query)
    ranked = search(index, query, scorer=make_scorer(scorer), k=k * page)[(page - 1) * k :]
    hits = [
        Hit(
            doc_id=result.doc_id,
            title=result.title,
            score=result.score,
            snippet=make_snippet(
                read_document(document_path(index, result.doc_id, corpus_dir)), terms, mark=_MARK
            ),
        )
        for result in ranked
    ]
    return Page(len(matched), (time.perf_counter() - started) * 1000, hits)


class SemanticUnavailableError(Exception):
    """Режим semantic/hybrid без файлу векторів."""


def run_search_mode(
    index: Index,
    corpus_dir: Path | None,
    query: str,
    k: int,
    page: int,
    scorer: ScorerName,
    mode: Mode,
    semantic: tuple[SemanticIndex, Embedder] | None,
) -> Page:
    """keyword — Лаби 1–8 без змін; semantic — лише вектори; hybrid — RRF обох списків."""
    if mode == "keyword":
        return run_search(index, corpus_dir, query, k, page, scorer)
    if semantic is None:
        raise SemanticUnavailableError("семантичний пошук вимкнено: немає файлу векторів")
    sem, embedder = semantic
    started = time.perf_counter()
    sem_hits = {h.doc_id: h for h in semantic_rank(sem, embedder, query)}
    rankings: list[list[int]] = [list(sem_hits)]
    terms: tuple[str, ...] = ()
    if mode == "hybrid":
        kw_query = keyword_query(query)
        if kw_query:
            terms, _ = matching_ids(index, kw_query)
            keyword = search_vectorized(index, kw_query, scorer, index.num_docs)
            rankings.append([r.doc_id for r in keyword])
    # semantic: показуємо справжній косинус; RRF з одного списку — це лише 1/(60+місце), без змісту
    fused = (
        [(h.doc_id, h.score) for h in sem_hits.values()] if mode == "semantic" else rrf(rankings)
    )
    window = fused[(page - 1) * k : page * k]
    titles = {m.doc_id: m.path.stem for m in index.docs()}
    hits: list[Hit] = []
    for doc_id, score in window:
        doc = load_doc_tokens(document_path(index, doc_id, corpus_dir))
        snippet = snippet_from_tokens(doc, terms, mark=_MARK) if terms else ""
        if not snippet and doc_id in sem_hits:
            snippet = chunk_text(sem, sem_hits[doc_id].chunk, doc.text)
        hits.append(Hit(doc_id, titles[doc_id], score, snippet))
    return Page(len(fused), (time.perf_counter() - started) * 1000, hits)


def plain_snippet(snippet: str) -> str:
    return snippet.replace(_MARK, "")


def highlight(snippet: str) -> Markup:
    parts = snippet.split(_MARK)
    return Markup("").join(
        Markup("<mark>{}</mark>").format(part) if i % 2 else Markup.escape(part)
        for i, part in enumerate(parts)
    )
