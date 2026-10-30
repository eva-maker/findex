"""Командний рядок findex: `findex index | search | stats`.

Правила цього модуля:
  * stdout — лише ДАНІ (таблиця результатів, JSON). Усе діагностичне — у stderr.
  * очікувана помилка (немає файлу, поганий запит) — один рядок і ненульовий код,
    без traceback.
  * logging налаштовується один раз, тут, у точці входу. Бібліотечні модулі
    лише пишуть у свої `logging.getLogger(__name__)`.
"""

import json
import logging
import sys
import time
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.text import Text

from findex.corpus import list_document_paths, load_document, read_document
from findex.index import index_documents
from findex.query import QueryError
from findex.ranking import ScorerName, make_scorer, matching_ids, search
from findex.serialize import IndexLoadError, open_index, save_index
from findex.snippets import make_snippet
from findex.stats import index_stats

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
    """typer будує --help і перевірку значень з Enum; нижче перетворюємо його в Literal."""

    bm25 = "bm25"
    tfidf = "tfidf"

    def as_name(self) -> ScorerName:
        return "bm25" if self is ScorerChoice.bm25 else "tfidf"


def configure_logging(verbosity: int) -> None:
    """0 -> WARNING, -v -> INFO, -vv -> DEBUG. Усе йде в stderr."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    package_logger = logging.getLogger("findex")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(level)


def fail(message: str) -> NoReturn:
    """Очікувана помилка: один зрозумілий рядок у stderr і код виходу 1."""
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
    corpus: Annotated[Path, typer.Argument(help="Каталог з .txt-документами.")],
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
) -> None:
    """Побудувати індекс з каталогу текстів і зберегти у файл."""
    if not corpus.is_dir():
        fail(f"каталог не знайдено: {corpus}")
    paths = list_document_paths(corpus)
    if not paths:
        fail(f"у каталозі {corpus} немає .txt-файлів")
    if limit is not None:
        paths = paths[:limit]

    start = time.perf_counter()
    # track() малює смугу прогресу в stderr і віддає елементи по одному:
    # документи, як і раніше, читаються стрімом, не всі одразу.
    docs = (
        load_document(path)
        for path in track(paths, description="Індексація", console=err, transient=True)
    )
    index = index_documents(docs, positions=positions)
    try:
        save_index(index, out_path)
    except OSError as error:
        fail(f"не вдалося записати {out_path}: {error.strerror or error}")

    elapsed = time.perf_counter() - start
    out.print(f"{index}  ->  {out_path}  ({elapsed:.1f} с)")


def _snippet_text(snippet: str) -> Text:
    """Рядок із _MARK навколо збігів -> rich Text із виділеними збігами."""
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
) -> None:
    """Ранжований пошук по збереженому індексу."""
    try:
        with open_index(index_path) as index:
            results = search(index, query, scorer=make_scorer(scorer.as_name()), k=k)
            terms, _matched = matching_ids(index, query)
            rows: list[tuple[int, float, int, Path, str]] = []
            for rank, result in enumerate(results, start=1):
                path = index.doc(result.doc_id).path
                mark = "" if as_json else _MARK
                snippet = make_snippet(read_document(path), terms, mark=mark)
                rows.append((rank, result.score, result.doc_id, path, snippet))
    except (IndexLoadError, QueryError) as error:
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
    table = Table(title=f"{query!r}  ({scorer.value})", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("Оцінка", justify="right")
    table.add_column("Документ")
    table.add_column("Фрагмент", overflow="fold")
    for rank, score, _doc_id, path, snippet in rows:
        table.add_row(str(rank), f"{score:.3f}", path.stem, _snippet_text(snippet))
    out.print(table)


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
