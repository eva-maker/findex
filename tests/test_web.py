"""Тести веб-шару: TestClient + dependency_overrides, без справжнього файлу індексу."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.models import Index
from findex.serialize import IndexLoadError, save_index
from findex.web.app import create_app
from findex.web.deps import get_index
from findex.web.settings import Settings


def make_settings(index_path: Path, corpus_dir: Path | None = None, **extra: bool) -> Settings:
    # _env_file=None: тест не повинен залежати від .env розробника
    return Settings(_env_file=None, index_path=index_path, corpus_dir=corpus_dir, **extra)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def client(tiny_index: Index, tiny_corpus: Path, tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(make_settings(tmp_path / "unused.bin", tiny_corpus))
    # Підміна залежності: маршрути отримують крихітний індекс, файл з диска не потрібен.
    app.dependency_overrides[get_index] = lambda: tiny_index
    yield TestClient(app)


def test_search_ok(client: TestClient) -> None:
    response = client.get("/search", params={"q": "cat"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "cat"
    assert body["total"] == 2  # a.txt і b.txt
    assert {hit["title"] for hit in body["results"]} == {"a", "b"}
    assert "cat" in body["results"][0]["snippet"]
    assert "\x00" not in body["results"][0]["snippet"]


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "cat", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "cat", "k": 1, "page": 2}).json()
    third = client.get("/search", params={"q": "cat", "k": 1, "page": 3}).json()
    assert first["total"] == second["total"] == 2
    assert len(first["results"]) == len(second["results"]) == 1
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    assert third["results"] == []


@pytest.mark.parametrize(
    "params",
    [
        {"q": "cat", "k": "abc"},
        {"q": "cat", "k": 0},
        {"q": "cat", "k": 101},
        {"q": "cat", "page": 0},
        {"q": "cat", "scorer": "nope"},
        {"q": ""},
        {},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, str | int]) -> None:
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_search_bad_query_400(client: TestClient) -> None:
    response = client.get("/search", params={"q": "cat AND ("})
    assert response.status_code == 400
    assert response.json()["detail"]


def test_document_ok_and_404(client: TestClient) -> None:
    ok = client.get("/docs/0")
    assert ok.status_code == 200
    assert ok.json()["title"] == "a"
    assert "cat" in ok.json()["text"]
    assert client.get("/docs/999").status_code == 404


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["doc_count"] == 3
    assert body["vocab_size"] > 0
    assert body["uptime_s"] >= 0
    assert body["top_terms"][0]["count"] >= body["top_terms"][-1]["count"]


def test_health_ok(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_health_503_when_index_not_loaded(tmp_path: Path) -> None:
    # Без `with` lifespan не запускається, індексу в app.state немає.
    app = create_app(make_settings(tmp_path / "unused.bin"))
    response = TestClient(app).get("/health")
    assert response.status_code == 503


def test_openapi_page(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)


def test_request_id_header(client: TestClient) -> None:
    assert len(client.get("/health").headers["X-Request-ID"]) == 8


def test_unexpected_error_is_clean_500(tiny_index: Index, tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "unused.bin", tmp_path / "no_such_dir"))
    app.dependency_overrides[get_index] = lambda: tiny_index
    # Каталог корпусу відсутній -> FileNotFoundError усередині маршруту: це «несподіваний» виняток.
    response = TestClient(app, raise_server_exceptions=False).get("/docs/0")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal Server Error"
    assert "Traceback" not in response.text
    assert len(body["request_id"]) == 8


def test_html_home_and_highlight(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    page = client.get("/", params={"q": "cat"})
    assert page.status_code == 200
    assert "<mark>cat</mark>" in page.text
    assert "/doc/0" in page.text or "/doc/1" in page.text


def test_html_bad_query_shows_error(client: TestClient) -> None:
    page = client.get("/", params={"q": "cat AND ("})
    assert page.status_code == 200
    assert "Не вдалося розібрати запит" in page.text


def test_html_document_page(client: TestClient) -> None:
    page = client.get("/doc/0")
    assert page.status_code == 200
    assert "The cat sat" in page.text


def test_lifespan_loads_index_once(tiny_index: Index, tiny_corpus: Path, tmp_path: Path) -> None:
    index_file = tmp_path / "index.bin"
    save_index(tiny_index, index_file)
    app = create_app(make_settings(index_file, tiny_corpus))
    with TestClient(app) as live:  # with => lifespan працює
        assert live.get("/health").json() == {"status": "ok"}
        assert live.get("/search", params={"q": "sea"}).json()["total"] == 1


def test_missing_index_fails_at_startup(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path / "no_such.bin"))
    with pytest.raises(IndexLoadError), TestClient(app):
        pass


def test_bench_routes_only_when_enabled(
    tiny_index: Index, tiny_corpus: Path, tmp_path: Path
) -> None:
    off = create_app(make_settings(tmp_path / "x.bin", tiny_corpus))
    on = create_app(make_settings(tmp_path / "x.bin", tiny_corpus, enable_bench_routes=True))
    for app in (off, on):
        app.dependency_overrides[get_index] = lambda: tiny_index
    assert TestClient(off).get("/_bench/search-async", params={"q": "cat"}).status_code == 404
    for path in ("/_bench/search-async", "/_bench/search-to-thread"):
        response = TestClient(on).get(path, params={"q": "cat"})
        assert response.json() == {"total": 2}


def test_settings_require_index_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INDEX_PATH", raising=False)
    with pytest.raises(ValueError, match="index_path"):
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
