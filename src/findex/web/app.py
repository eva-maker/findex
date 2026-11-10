"""FastAPI-застосунок: lifespan, маршрути JSON і HTML, обробка помилок.

Запуск: `findex serve` (або `uvicorn findex.web.app:create_app --factory`).
"""

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex.corpus import read_document
from findex.models import Index
from findex.query import QueryError
from findex.ranking import ScorerName
from findex.serialize import load_index
from findex.snippets import warm_doc_tokens
from findex.stats import IndexStats, index_stats
from findex.vectorized import vectors_for
from findex.web import bench_routes
from findex.web.deps import IndexDep, SettingsDep
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    StatsResponse,
    TermCount,
)
from findex.web.service import document_path, highlight, plain_snippet, run_search
from findex.web.settings import Settings

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["highlight"] = highlight


@lru_cache(maxsize=1)
def _stats(index: Index) -> IndexStats:
    """Статистика рахується один раз на індекс: обхід усього індексу задорогий для кожного запиту."""
    return index_stats(index, top=10)


def create_app(settings: Settings | None = None) -> FastAPI:
    # Settings() без аргументів читає INDEX_PATH з оточення; немає — ValidationError одразу.
    settings = settings or Settings()  # pyright: ignore[reportCallIssue]
    logging.getLogger("findex").setLevel(settings.log_level.upper())
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:

        app.state.index = load_index(settings.index_path)
        app.state.started_at = time.monotonic()
        log.info("індекс завантажено: %s", app.state.index)
        vectors_for(app.state.index)
        warm_started = time.perf_counter()
        warmed = warm_doc_tokens(
            document_path(app.state.index, meta.doc_id, settings.corpus_dir)
            for meta in app.state.index.docs()
        )
        log.info("токени %d документів: %.1f с", warmed, time.perf_counter() - warm_started)
        yield
        app.state.index.close()  # завершення: SIGTERM -> uvicorn -> цей рядок

    app = FastAPI(title="findex", version="0.7.0", lifespan=lifespan)
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    if settings.enable_bench_routes:
        app.include_router(bench_routes.router)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = uuid.uuid4().hex[:8]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:

        rid = getattr(request.state, "request_id", "-")
        log.error("необроблений виняток [%s] %s", rid, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "Internal Server Error", "request_id": rid}, status_code=500)

    @app.get("/search", response_model=SearchResponse)
    def search(
        index: IndexDep,
        settings: SettingsDep,
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        scorer: ScorerName = "bm25",
    ) -> SearchResponse:
        try:
            result = run_search(index, settings.corpus_dir, q, k, page, scorer)
        except QueryError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return SearchResponse(
            query=q,
            total=result.total,
            page=page,
            k=k,
            took_ms=round(result.took_ms, 2),
            results=[
                SearchHit(
                    doc_id=h.doc_id,
                    title=h.title,
                    score=round(h.score, 6),
                    snippet=plain_snippet(h.snippet),
                )
                for h in result.hits
            ],
        )

    @app.get("/docs/{doc_id}", response_model=DocumentResponse)
    def get_document(doc_id: int, index: IndexDep, settings: SettingsDep) -> DocumentResponse:
        if doc_id not in index.doc_ids():
            raise HTTPException(status_code=404, detail=f"документа {doc_id} немає")
        path = document_path(index, doc_id, settings.corpus_dir)
        return DocumentResponse(
            doc_id=doc_id,
            title=index.doc(doc_id).path.stem,
            length=index.doc_length(doc_id),
            text=read_document(path),
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats(index: IndexDep, request: Request) -> StatsResponse:
        s = _stats(index)
        started: float = getattr(request.app.state, "started_at", time.monotonic())
        return StatsResponse(
            doc_count=s["doc_count"],
            token_count=s["token_count"],
            vocab_size=s["vocab_size"],
            avg_doc_length=s["avg_doc_length"],
            has_positions=s["has_positions"],
            top_terms=[TermCount(term=t, count=c) for t, c in s["top_terms"]],
            uptime_s=round(time.monotonic() - started, 1),
        )

    @app.get("/health", response_model=HealthResponse)
    def health(index: IndexDep) -> HealthResponse:
        # Сам факт, що get_index віддав індекс, і є перевіркою; немає індексу — 503 з Depends.
        return HealthResponse(status="ok" if len(index) > 0 else "empty")

    @app.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        index: IndexDep,
        settings: SettingsDep,
        q: str = "",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        scorer: ScorerName = "bm25",
    ) -> HTMLResponse:
        per_page = 10
        context: dict[str, object] = {"q": q, "scorer": scorer, "page": page, "searched": False}
        if q.strip():
            context["searched"] = True
            try:
                result = run_search(index, settings.corpus_dir, q, per_page, page, scorer)
            except QueryError as error:
                context["error"] = str(error)
            else:
                context.update(
                    total=result.total,
                    took_ms=round(result.took_ms, 1),
                    hits=result.hits,
                    has_next=page * per_page < result.total,
                )
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def doc_page(
        request: Request, doc_id: int, index: IndexDep, settings: SettingsDep, q: str = ""
    ) -> HTMLResponse:
        if doc_id not in index.doc_ids():
            raise HTTPException(status_code=404, detail=f"документа {doc_id} немає")
        text = read_document(document_path(index, doc_id, settings.corpus_dir))
        context = {"doc_id": doc_id, "title": index.doc(doc_id).path.stem, "text": text, "q": q}
        return templates.TemplateResponse(request, "doc.html", context)

    return app
