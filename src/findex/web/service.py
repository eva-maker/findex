import time
from dataclasses import dataclass
from pathlib import Path

from markupsafe import Markup

from findex.corpus import read_document
from findex.models import Index
from findex.ranking import ScorerName, make_scorer, matching_ids, search
from findex.snippets import load_doc_tokens, make_snippet, snippet_from_tokens
from findex.vectorized import search_vectorized

_MARK = "\x00"


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


def plain_snippet(snippet: str) -> str:
    return snippet.replace(_MARK, "")


def highlight(snippet: str) -> Markup:
    parts = snippet.split(_MARK)
    return Markup("").join(
        Markup("<mark>{}</mark>").format(part) if i % 2 else Markup.escape(part)
        for i, part in enumerate(parts)
    )
