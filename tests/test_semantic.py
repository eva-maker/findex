"""Семантичний пошук без справжньої моделі: «підроблений» embedder зі словником значень.

Так тести швидкі, детерміновані й не качають 90 МБ; перевіряємо нашу логіку
(шматки, косинус, RRF, API), а не якість all-MiniLM-L6-v2 — її міряє evaluate_modes.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from findex.models import Index
from findex.semantic import (
    Matrix,
    SemanticError,
    SemanticIndex,
    build_semantic_index,
    chunk_spans,
    keyword_query,
    rrf,
    semantic_rank,
)
from findex.serialize import save_index
from findex.tokens import tokenize
from findex.web.app import create_app
from findex.web.settings import Settings

# Слова-синоніми мають спільну «координату» — як близькі слова в справжньої моделі.
CONCEPTS = {"cat": 0, "feline": 0, "kitten": 0, "dog": 1, "canine": 1, "sea": 2, "ocean": 2}


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in tokenize(text):
                if word in CONCEPTS:
                    out[row, CONCEPTS[word]] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


@pytest.fixture
def sem(tiny_index: Index, tiny_corpus: Path) -> SemanticIndex:
    return build_semantic_index(tiny_index, tiny_corpus, FakeEmbedder(), model="fake")


def test_chunk_spans_cover_text_in_pieces() -> None:
    text = "one two three four five six seven"
    spans = chunk_spans(text, words=3)
    assert [text[s:e] for s, e in spans] == ["one two three", "four five six", "seven"]
    assert chunk_spans("", words=3) == []


def test_semantic_finds_document_without_shared_words(sem: SemanticIndex) -> None:
    # У жодному документі немає слова «feline» чи «ocean» — лише синоніми.
    assert [h.doc_id for h in semantic_rank(sem, FakeEmbedder(), "feline")][:2] == [0, 1]
    assert semantic_rank(sem, FakeEmbedder(), "ocean")[0].doc_id == 2


def test_semantic_rank_limit_and_unknown_words(sem: SemanticIndex) -> None:
    assert len(semantic_rank(sem, FakeEmbedder(), "feline", k=1)) == 1
    # запит без відомих моделі слів: нульовий вектор, але без падіння й NaN
    hits = semantic_rank(sem, FakeEmbedder(), "zzz")
    assert len(hits) == 3
    assert all(h.score == 0.0 for h in hits)


def test_save_load_roundtrip(sem: SemanticIndex, tmp_path: Path) -> None:
    path = tmp_path / "e.npz"
    sem.save(path)
    loaded = SemanticIndex.load(path)
    assert loaded.model == "fake"
    assert np.array_equal(loaded.vectors, sem.vectors)
    assert np.array_equal(loaded.chunk_doc, sem.chunk_doc)


def test_load_broken_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    with pytest.raises(SemanticError):
        SemanticIndex.load(bad)
    with pytest.raises(SemanticError):
        SemanticIndex.load(tmp_path / "missing.npz")


def test_rrf_values_and_tie_break() -> None:
    fused = dict(rrf([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    # 1 і 3 мають різні оцінки; рівні оцінки — за doc_id
    assert [d for d, _ in rrf([[5, 4]])] == [5, 4]
    assert [d for d, _ in rrf([[5], [4]])] == [4, 5]


def test_keyword_query() -> None:
    assert keyword_query("old man sea") == "old OR man OR sea"
    assert keyword_query("cat cat dog") == "cat OR dog"
    assert keyword_query('"old man" AND sea') == '"old man" AND sea'
    assert keyword_query("!!!") == ""


def make_settings(tmp_path: Path, corpus: Path, **extra: Path) -> Settings:
    # _env_file=None: тест не залежить від .env розробника
    return Settings(_env_file=None, index_path=tmp_path / "x.bin", corpus_dir=corpus, **extra)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def client(
    tiny_index: Index, tiny_corpus: Path, sem: SemanticIndex, tmp_path: Path
) -> Iterator[TestClient]:
    save_index(tiny_index, tmp_path / "x.bin")
    sem.save(tmp_path / "e.npz")
    settings = make_settings(tmp_path, tiny_corpus, embeddings_path=tmp_path / "e.npz")
    app = create_app(settings, embedder_factory=FakeEmbedder)
    with TestClient(app) as test_client:  # with: щоб відпрацював lifespan (завантаження)
        yield test_client


def test_api_semantic_mode(client: TestClient) -> None:
    body = client.get("/search", params={"q": "feline", "mode": "semantic"}).json()
    assert body["mode"] == "semantic"
    assert body["results"][0]["title"] in {"a", "b"}
    assert body["results"][0]["snippet"]  # сніпет — найкращий шматок


def test_api_semantic_scores_are_cosines(client: TestClient) -> None:
    scores = [
        h["score"]
        for h in client.get("/search", params={"q": "feline", "mode": "semantic"}).json()["results"]
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)  # документ a: самі «кішки»
    assert scores[1] == pytest.approx(0.707107, abs=1e-5)  # b: кішка + собака


def test_api_hybrid_mode_ranks_both_signals(client: TestClient) -> None:
    body = client.get("/search", params={"q": "sea ocean", "mode": "hybrid"}).json()
    assert body["results"][0]["title"] == "c"
    assert "\x00" not in body["results"][0]["snippet"]


def test_api_keyword_mode_unchanged(client: TestClient) -> None:
    body = client.get("/search", params={"q": "cat"}).json()
    assert body["mode"] == "keyword"
    assert {h["title"] for h in body["results"]} == {"a", "b"}


def test_api_bad_mode(client: TestClient) -> None:
    assert client.get("/search", params={"q": "cat", "mode": "magic"}).status_code == 422


def test_api_semantic_without_embeddings_is_400(
    tiny_index: Index, tiny_corpus: Path, tmp_path: Path
) -> None:
    save_index(tiny_index, tmp_path / "x.bin")
    with TestClient(create_app(make_settings(tmp_path, tiny_corpus))) as test_client:
        response = test_client.get("/search", params={"q": "cat", "mode": "semantic"})
        assert response.status_code == 400
        assert "вектор" in response.json()["detail"]
        assert 'name="mode"' not in test_client.get("/").text  # перемикача без векторів немає


def test_home_page_with_vectors(client: TestClient) -> None:
    assert 'name="mode"' in client.get("/").text
    page = client.get("/", params={"q": "feline", "mode": "semantic"})
    assert page.status_code == 200
    assert "<article>" in page.text
