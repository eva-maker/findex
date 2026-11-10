import asyncio
import json
import logging
import os
import sys
import time
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Final, NoReturn

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from findex.corpus import list_document_paths, read_document, unpack_jsonl
from findex.crawler.crawl import DEFAULT_CONTACT, CrawlStats, make_user_agent
from findex.crawler.display import render_stats
from findex.crawler.runner import CrawlSettings, run_crawl
from findex.crawler.urls import canonicalize
from findex.index import Chunk, ExecutorName, build_index_parallel
from findex.query import QueryError
from findex.ranking import ScorerName, make_scorer, matching_ids, search
from findex.semantic import (
    DEFAULT_MODEL,
    FastEmbedder,
    SemanticError,
    SemanticIndex,
    build_semantic_index,
)
from findex.serialize import IndexLoadError, open_index, save_index
from findex.snippets import make_snippet
from findex.stats import index_stats
from findex.web.service import SemanticUnavailableError, run_search_mode

log = logging.getLogger(__name__)

app = typer.Typer(
    help="findex — пошук по корпусу текстів: побудова індексу, ранжований пошук, статистика.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

# Дані йдуть у stdout, діагностика й прогрес — у stderr.
out = Console()
err = Console(stderr=True)

# Маркер виділення в сніпеті. Беремо символ, якого в текстах не буває, щоб він
# не плутався зі звичайними зірочками (Gutenberg їх використовує).
_MARK = "\x00"


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"

    def as_name(self) -> ScorerName:
        return "bm25" if self is ScorerChoice.bm25 else "tfidf"


class ModeChoice(StrEnum):
    keyword = "keyword"
    semantic = "semantic"
    hybrid = "hybrid"


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"

    def as_name(self) -> ExecutorName:
        match self:
            case ExecutorChoice.serial:
                return "serial"
            case ExecutorChoice.threads:
                return "threads"
            case ExecutorChoice.processes:
                return "processes"


DEFAULT_WORKERS_CAP: Final = 4


def default_workers() -> int:
    return min(DEFAULT_WORKERS_CAP, os.cpu_count() or 1)


def configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    package_logger = logging.getLogger("findex")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(level)


def fail(message: str) -> NoReturn:
    typer.echo(f"Помилка: {message}", err=True)
    raise typer.Exit(code=1)


def _version_callback(show: bool) -> None:
    if show:
        try:
            typer.echo(f"findex {version('findex')}")
        except PackageNotFoundError:
            typer.echo("findex (версія невідома: пакет не встановлено)")
        raise typer.Exit()


@app.callback()
def root(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="-v — INFO, -vv — DEBUG (логи йдуть у stderr).",
        ),
    ] = 0,
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Показати версію.",
        ),
    ] = False,
) -> None:
    configure_logging(verbose)


@app.command("index")
def index_command(
    corpus: Annotated[
        Path,
        typer.Argument(help="Каталог з .txt-документами або .jsonl, який зібрав findex crawl."),
    ],
    out_path: Annotated[
        Path,
        typer.Option("--out", "-o", help="Куди зберегти індекс (.json — JSON, інакше pickle)."),
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Зберігати позиції слів (потрібні для фраз)."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Взяти лише перші N документів."),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Скільки потоків чи процесів (за замовч. min(4, ядер); для serial не потрібно).",
        ),
    ] = None,
    executor: Annotated[
        ExecutorChoice,
        typer.Option(
            "--executor", help="serial — один цикл, threads — потоки, processes — процеси."
        ),
    ] = ExecutorChoice.processes,
    chunks: Annotated[
        int | None,
        typer.Option(
            "--chunks", min=1, help="На скільки шматків ділити корпус (за замовч. = workers)."
        ),
    ] = None,
    pages_dir: Annotated[
        Path | None,
        typer.Option("--pages-dir", help="Куди розпакувати сторінки з .jsonl (за замовч. поруч)."),
    ] = None,
) -> None:
    """Побудувати індекс з каталогу текстів (або з JSONL краулера) і зберегти у файл."""
    if corpus.is_file() and corpus.suffix == ".jsonl":
        # Індексація йде поза циклом подій і поза краулером: спочатку розкладаємо сторінки по
        # .txt (так працюють і пошук, і сніпети), далі все як у Лабі 5.
        target = pages_dir if pages_dir is not None else corpus.with_name(corpus.stem + "_pages")
        count = unpack_jsonl(corpus, target)
        err.print(f"Розпаковано {count} сторінок з {corpus} у {target}")
        corpus = target
    if not corpus.is_dir():
        fail(f"каталог не знайдено: {corpus}")
    paths = list_document_paths(corpus)
    if not paths:
        fail(f"у каталозі {corpus} немає .txt-файлів")
    if limit is not None:
        paths = paths[:limit]

    if executor is ExecutorChoice.serial and workers is not None and workers > 1:
        log.warning("--workers %d ігнорується: executor serial працює в одному циклі", workers)
    worker_count = workers if workers is not None else default_workers()

    start = time.perf_counter()
    # Смуга прогресу в stderr: рух на кожен готовий шматок (файли кожного воркер читає сам).
    with Progress(console=err, transient=True) as progress:
        task = progress.add_task("Індексація", total=len(paths))

        def advance(chunk: Chunk) -> None:
            progress.advance(task, len(chunk.paths))

        index = build_index_parallel(
            paths,
            positions=positions,
            workers=worker_count,
            executor=executor.as_name(),
            chunks=chunks,
            on_chunk_done=advance,
        )
    try:
        save_index(index, out_path)
    except OSError as error:
        fail(f"не вдалося записати {out_path}: {error.strerror or error}")

    elapsed = time.perf_counter() - start
    out.print(f"{index}  ->  {out_path}  ({elapsed:.1f} с)")


def _snippet_text(snippet: str) -> Text:
    text = Text()
    for i, part in enumerate(snippet.split(_MARK)):
        text.append(part, style="bold yellow" if i % 2 else None)
    return text


@app.command("search")
def search_command(
    index_path: Annotated[Path, typer.Argument(help="Файл індексу.")],
    query: Annotated[str, typer.Argument(help='Запит, напр. "war AND NOT peace" або "old man".')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Скільки результатів показати.")] = 10,
    scorer: Annotated[
        ScorerChoice, typer.Option("--scorer", help="Формула релевантності.")
    ] = ScorerChoice.bm25,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="JSON-рядки в stdout (для пайпів) замість таблиці."),
    ] = False,
    mode: Annotated[
        ModeChoice, typer.Option("--mode", help="keyword — за словами, semantic — за змістом.")
    ] = ModeChoice.keyword,
    embeddings: Annotated[
        Path,
        typer.Option("--embeddings", help="Файл векторів (створює `findex embed`)."),
    ] = Path("embeddings.npz"),
) -> None:
    """Ранжований пошук по збереженому індексу."""
    try:
        with open_index(index_path) as index:
            rows: list[tuple[int, float, int, Path, str]] = []
            if mode is ModeChoice.keyword:
                results = search(index, query, scorer=make_scorer(scorer.as_name()), k=k)
                terms, _matched = matching_ids(index, query)
                for rank, result in enumerate(results, start=1):
                    path = index.doc(result.doc_id).path
                    mark = "" if as_json else _MARK
                    snippet = make_snippet(read_document(path), terms, mark=mark)
                    rows.append((rank, result.score, result.doc_id, path, snippet))
            else:
                sem = SemanticIndex.load(embeddings)
                pair = (sem, FastEmbedder(sem.model))
                page = run_search_mode(index, None, query, k, 1, scorer.as_name(), mode.value, pair)
                for rank, hit in enumerate(page.hits, start=1):
                    snippet = hit.snippet.replace(_MARK, "") if as_json else hit.snippet
                    rows.append((rank, hit.score, hit.doc_id, index.doc(hit.doc_id).path, snippet))
    except (IndexLoadError, QueryError, SemanticError, SemanticUnavailableError) as error:
        fail(str(error))
    except OSError as error:
        fail(f"не вдалося прочитати файл: {error.filename or error}")

    if as_json:
        for rank, score, doc_id, path, snippet in rows:
            record = {
                "rank": rank,
                "score": round(score, 6),
                "doc_id": doc_id,
                "title": path.stem,
                "path": str(path),
                "snippet": snippet,
            }
            typer.echo(
                json.dumps(record)
            )  # ensure_ascii за замовчуванням: безпечно для будь-якого терміналу
        return

    if not rows:
        log.info("за запитом %r нічого не знайдено", query)
        out.print("Нічого не знайдено.")
        return
    label = scorer.value if mode is ModeChoice.keyword else mode.value
    table = Table(title=f"{query!r}  ({label})", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Оцінка", justify="right")
    table.add_column("Документ")
    table.add_column("Фрагмент", overflow="fold")
    for rank, score, _doc_id, path, snippet in rows:
        table.add_row(str(rank), f"{score:.3f}", path.stem, _snippet_text(snippet))
    out.print(table)


@app.command("embed")
def embed_command(
    index_path: Annotated[Path, typer.Argument(help="Файл індексу.")],
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", help="Каталог з .txt (якщо шляхи в індексі застаріли)."),
    ] = None,
    out_path: Annotated[
        Path, typer.Option("--out", "-o", help="Куди зберегти вектори (.npz).")
    ] = Path("embeddings.npz"),
    model: Annotated[str, typer.Option("--model", help="Модель fastembed.")] = DEFAULT_MODEL,
    cache: Annotated[
        Path | None, typer.Option("--cache", help="Де зберігати завантажену модель.")
    ] = None,
) -> None:
    """Порахувати вектори шматків тексту для семантичного пошуку (повільно, раз на корпус)."""
    start = time.perf_counter()
    try:
        with open_index(index_path) as index:
            err.print(f"Завантажую модель {model} (перший запуск качає ~90 МБ)…")
            embedder = FastEmbedder(model, cache)
            sem = build_semantic_index(index, corpus, embedder, model=model)
        sem.save(out_path)
    except (IndexLoadError, SemanticError) as error:
        fail(str(error))
    except OSError as error:
        fail(f"не вдалося прочитати/записати файл: {error.filename or error}")
    size = out_path.stat().st_size / 2**20
    out.print(
        f"{len(sem)} шматків -> {out_path} ({size:.1f} МБ, {time.perf_counter() - start:.0f} с)"
    )


@app.command("stats")
def stats_command(
    index_path: Annotated[Path, typer.Argument(help="Файл індексу.")],
    top: Annotated[
        int, typer.Option("--top", min=1, help="Скільки частих термінів показати.")
    ] = 10,
) -> None:
    """Показати основні числа індексу (як у Лабі 1)."""
    try:
        with open_index(index_path) as index:
            stats = index_stats(index, top=top)
    except IndexLoadError as error:
        fail(str(error))

    summary = Table(show_header=False, box=None)
    summary.add_row("Документів", f"{stats['doc_count']:,}")
    summary.add_row("Токенів", f"{stats['token_count']:,}")
    summary.add_row("Унікальних термінів", f"{stats['vocab_size']:,}")
    summary.add_row("Середня довжина документа", f"{stats['avg_doc_length']:,.0f}")
    summary.add_row("Позиції слів", "так" if stats["has_positions"] else "ні")
    out.print(summary)

    top_table = Table(title=f"Топ-{top} термінів")
    top_table.add_column("Термін")
    top_table.add_column("Входжень", justify="right")
    for term, count in stats["top_terms"]:
        top_table.add_row(term, f"{count:,}")
    out.print(top_table)


# Підміна транспорту для тестів: у справжньому запуску лишається None (справжня мережа).
_test_transport: httpx.AsyncBaseTransport | None = None


def _attach_crawl_log(path: Path, debug: bool) -> logging.Handler:
    """crawl.log: статус кожного URL (логер findex.crawler.access) + повтори й попередження.

    Із --debug сюди ж потрапляють скарги asyncio на колбеки, що блокували цикл (>100 мс).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    for name in ("findex.crawler", "asyncio") if debug else ("findex.crawler",):
        target = logging.getLogger(name)
        target.addHandler(handler)
        target.setLevel(logging.DEBUG if name == "asyncio" else logging.INFO)
    logging.getLogger("findex.crawler").propagate = False  # не заважати живому екрану в stderr
    return handler


def _detach_crawl_log(handler: logging.Handler) -> None:
    for name in ("findex.crawler", "asyncio"):
        logging.getLogger(name).removeHandler(handler)
    logging.getLogger("findex.crawler").propagate = True
    handler.close()


@app.command("crawl")
def crawl_command(
    seed: Annotated[str, typer.Argument(help="Початкова адреса, напр. http://127.0.0.1:8000/.")],
    max_pages: Annotated[
        int, typer.Option("--max-pages", min=1, help="Стеля на кількість сторінок.")
    ] = 500,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Скільки запитів одночасно.")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", min=1, help="Скільки запитів одночасно до одного хоста.")
    ] = 2,
    out_path: Annotated[Path, typer.Option("--out", "-o", help="JSONL-файл зі сторінками.")] = Path(
        "data/crawl.jsonl"
    ),
    delay: Annotated[
        float, typer.Option("--delay", min=0.0, help="Мінімальна пауза між запитами до хоста, с.")
    ] = 0.25,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1, help="Таймаут запиту, с.")] = 10.0,
    domain: Annotated[
        list[str] | None,
        typer.Option("--domain", help="Дозволений домен (можна кілька; за замовч. хост seed)."),
    ] = None,
    log_file: Annotated[
        Path, typer.Option("--log-file", help="Файл зі статусом кожного URL.")
    ] = Path("crawl.log"),
    contact: Annotated[
        str, typer.Option("--contact", help="Адреса для зв'язку в User-Agent.")
    ] = DEFAULT_CONTACT,
    debug: Annotated[
        bool, typer.Option("--debug", help="asyncio debug: попереджає про блокування циклу.")
    ] = False,
) -> None:
    """Обійти сайт асинхронно (ввічливо) і дописувати сторінки у JSONL по мірі надходження."""
    if canonicalize(seed) is None:
        fail(f"це не http(s)-адреса: {seed}")
    settings = CrawlSettings(
        seeds=(seed,),
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=per_host,
        allowed_domains=domain,
        delay=delay,
        timeout=timeout,
        user_agent=make_user_agent(contact),
    )
    stats = CrawlStats()
    handler = _attach_crawl_log(log_file, debug)
    written = 0
    try:
        with Live(
            render_stats(stats, max_pages),
            console=err,
            refresh_per_second=4,
            transient=True,  # після обходу екран зникає, підсумок — у рядку нижче
            get_renderable=lambda: render_stats(stats, max_pages),
        ):
            written = asyncio.run(
                run_crawl(settings, out_path, stats, transport=_test_transport), debug=debug
            )
    except KeyboardInterrupt:
        err.print(f"Перервано. У {out_path} лишилось {stats.fetched} сторінок (запис — по рядку).")
        raise typer.Exit(code=130) from None
    except OSError as error:
        fail(f"не вдалося записати {out_path}: {error.strerror or error}")
    finally:
        _detach_crawl_log(handler)

    out.print(
        f"Зібрано {written} сторінок за {stats.elapsed:.1f} с ({stats.pages_per_second:.1f} стор/с), "
        f"помилок: {stats.errors}, повторів: {stats.retries}  ->  {out_path}  (журнал: {log_file})"
    )


@app.command("serve")
def serve_command(
    index_path: Annotated[
        Path | None,
        typer.Option("--index", help="Файл індексу (або змінна оточення INDEX_PATH)."),
    ] = None,
    corpus_dir: Annotated[
        Path | None,
        typer.Option("--corpus", help="Каталог з .txt для сніпетів (або CORPUS_DIR)."),
    ] = None,
    host: Annotated[
        str, typer.Option("--host", help="0.0.0.0 — слухати зовнішню мережу.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="Скільки процесів uvicorn (Лаба 5).")
    ] = 1,
) -> None:
    """Запустити веб-сервер: API на /docs, сторінка пошуку на /."""
    import uvicorn  # важкий імпорт: потрібен лише цій команді

    # Налаштування передаємо через оточення: так їх бачать і дочірні процеси --workers.
    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)
    if corpus_dir is not None:
        os.environ["CORPUS_DIR"] = str(corpus_dir)
    if "INDEX_PATH" not in os.environ and not Path(".env").is_file():
        fail("вкажіть --index або змінну оточення INDEX_PATH")
    # factory=True: uvicorn сам викличе create_app() у кожному процесі.
    uvicorn.run("findex.web.app:create_app", factory=True, host=host, port=port, workers=workers)
