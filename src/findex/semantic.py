"""Семантичний і гібридний пошук (Лаба 8, M3).

Ідея: кожен шматок тексту (~150 слів) перетворюємо на вектор-«відбиток змісту».
Запит теж стає вектором; близькі за змістом тексти мають близькі вектори,
навіть якщо спільних слів у них немає («car» ~ «automobile»).

Вектори нормуємо до довжини 1, тому косинусна схожість = звичайний добуток:
оцінки всіх шматків одним рядком `vectors @ query_vec` (матриця n×384 на вектор 384).

Гібрид зливає два списки (BM25 і семантичний) через Reciprocal Rank Fusion:
оцінка = Σ 1/(k + місце у списку). Складати самі оцінки не можна — BM25 і косинус
живуть у різних шкалах, а місця порівнювані.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from findex.corpus import read_document
from findex.models import Index
from findex.tokens import normalize_text, tokenize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 150
RRF_K = 60
_WORD_RE = re.compile(r"\S+")
_OPERATORS = re.compile(r'\b(?:AND|OR|NOT)\b|["()]')

Matrix = NDArray[np.float32]


class SemanticError(Exception):
    """Семантичний пошук недоступний або файл векторів зіпсований."""


class Embedder(Protocol):
    """Усе, що вміє перетворити тексти на нормовані вектори float32 [n, d]."""

    def embed(self, texts: Sequence[str]) -> Matrix: ...


def _normalize_rows(matrix: NDArray[np.floating]) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class FastEmbedder:
    """all-MiniLM-L6-v2 через ONNX (fastembed): без PyTorch, образ у рази легший."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - залежить від оточення
            raise SemanticError("встанови fastembed: uv add fastembed") from error
        self.model_name = model
        self._model = TextEmbedding(
            model_name=model, cache_dir=str(cache_dir) if cache_dir else None
        )

    def embed(self, texts: Sequence[str]) -> Matrix:
        vectors = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        return _normalize_rows(vectors)


def chunk_spans(text: str, words: int = CHUNK_WORDS) -> list[tuple[int, int]]:
    """Нарізати текст на шматки по `words` слів. Повертає символьні межі (start, end)."""
    spans = [m.span() for m in _WORD_RE.finditer(text)]
    return [
        (spans[i][0], spans[min(i + words, len(spans)) - 1][1]) for i in range(0, len(spans), words)
    ]


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    vectors: Matrix  # [n_chunks, dim], рядки нормовані
    chunk_doc: NDArray[np.int32]  # doc_id кожного шматка
    starts: NDArray[np.int32]  # межі шматка в normalize_text(read_document(...))
    ends: NDArray[np.int32]
    model: str

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            np.savez(
                handle,
                vectors=self.vectors,
                chunk_doc=self.chunk_doc,
                starts=self.starts,
                ends=self.ends,
                model=np.array(self.model),
            )

    @staticmethod
    def load(path: Path) -> SemanticIndex:
        try:
            with np.load(path) as data:
                return SemanticIndex(
                    vectors=data["vectors"].astype(np.float32, copy=False),
                    chunk_doc=data["chunk_doc"].astype(np.int32, copy=False),
                    starts=data["starts"].astype(np.int32, copy=False),
                    ends=data["ends"].astype(np.int32, copy=False),
                    model=str(data["model"]),
                )
        except (OSError, KeyError, ValueError) as error:
            raise SemanticError(f"не вдалося прочитати вектори {path}: {error}") from error


def build_semantic_index(
    index: Index,
    corpus_dir: Path | None,
    embedder: Embedder,
    model: str = DEFAULT_MODEL,
    words: int = CHUNK_WORDS,
    batch: int = 64,
) -> SemanticIndex:
    texts: list[str] = []
    docs: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for meta in index.docs():
        path = corpus_dir / meta.path.name if corpus_dir is not None else meta.path
        text = normalize_text(read_document(path))
        for start, end in chunk_spans(text, words):
            texts.append(text[start:end])
            docs.append(meta.doc_id)
            starts.append(start)
            ends.append(end)
    if not texts:
        raise SemanticError("у корпусі немає тексту для векторів")
    parts = [embedder.embed(texts[i : i + batch]) for i in range(0, len(texts), batch)]
    return SemanticIndex(
        vectors=np.vstack(parts),
        chunk_doc=np.asarray(docs, dtype=np.int32),
        starts=np.asarray(starts, dtype=np.int32),
        ends=np.asarray(ends, dtype=np.int32),
        model=model,
    )


@dataclass(frozen=True, slots=True)
class SemanticHit:
    doc_id: int
    score: float
    chunk: int  # індекс найкращого шматка документа


def semantic_rank(
    sem: SemanticIndex, embedder: Embedder, query: str, k: int | None = None
) -> list[SemanticHit]:
    """Документи за схожістю їхнього найкращого шматка до запиту (найкращі — першими)."""
    query_vec = embedder.embed([query])[0]
    sims = sem.vectors @ query_vec  # косинус: вектори вже нормовані
    best: dict[int, SemanticHit] = {}
    # від найсхожішого шматка: перша зустріч документа і є його найкращий шматок
    for chunk in np.argsort(-sims, kind="stable").tolist():
        doc_id = int(sem.chunk_doc[chunk])
        if doc_id not in best:
            best[doc_id] = SemanticHit(doc_id, float(sims[chunk]), chunk)
            if k is not None and len(best) >= k:
                break
    return list(best.values())


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Рівні оцінки впорядковуємо за doc_id (детермінізм)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def keyword_query(query: str) -> str:
    """Запит природною мовою -> OR із його слів, щоб BM25-частина гібрида не вимагала AND.

    Якщо людина сама написала оператори, лапки чи дужки — лишаємо як є."""
    if _OPERATORS.search(query):
        return query
    return " OR ".join(dict.fromkeys(tokenize(query)))


def chunk_text(sem: SemanticIndex, chunk: int, text: str, limit: int = 240) -> str:
    """Текст найкращого шматка (для сніпета), обрізаний до `limit` символів."""
    piece = text[int(sem.starts[chunk]) : int(sem.ends[chunk])]
    return piece if len(piece) <= limit else piece[:limit].rsplit(" ", 1)[0] + "…"
