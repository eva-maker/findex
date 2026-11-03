"""findex crawl: стрімінговий запис JSONL, crawl.log, зв'язок з iter_documents і findex index."""

import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from findex import cli
from findex.cli import app
from findex.corpus import iter_documents, iter_jsonl_documents, page_filename, unpack_jsonl
from findex.crawler.crawl import CrawlStats
from findex.crawler.display import render_stats
from findex.crawler.runner import CrawlSettings, run_crawl

runner = CliRunner()


def page_html(title: str, body: str, *links: str) -> str:
    anchors = " ".join(f'<a href="{link}">go</a>' for link in links)
    return f"<html><head><title>{title}</title></head><body><p>{body}</p>{anchors}</body></html>"


SITE = {
    "/": page_html("Головна", "Привіт, світе! zebra", "/a", "/b"),
    "/a": page_html("Alpha", "alpha apple banana"),
    "/b": page_html("Beta", "beta banana cherry"),
}


def site_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = SITE.get(request.url.path)
        return httpx.Response(200, html=body) if body is not None else httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    for name in ("findex", "findex.crawler"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def test_run_crawl_writes_one_json_line_per_page(tmp_path: Path) -> None:
    out = tmp_path / "pages.jsonl"
    stats = CrawlStats()
    settings = CrawlSettings(seeds=("http://e.com/",), max_pages=10, delay=0.0)
    written = asyncio.run(run_crawl(settings, out, stats, transport=site_transport()))
    assert written == 3 == stats.fetched
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["url"] for r in records} == {"http://e.com/", "http://e.com/a", "http://e.com/b"}
    assert set(records[0]) == {"url", "title", "text", "fetched_at"}
    home = next(r for r in records if r["url"] == "http://e.com/")
    assert home["title"] == "Головна"
    assert "Привіт" in out.read_text(encoding="utf-8")  # кирилиця не перетворена на \uXXXX


def test_lines_appear_while_the_crawl_is_still_running(tmp_path: Path) -> None:
    out = tmp_path / "pages.jsonl"
    seen_lines_during_crawl: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/":
            return httpx.Response(200, html=page_html("Home", "x", "/slow"))
        await asyncio.sleep(
            0.2
        )  # поки «повільна» сторінка вантажиться, головна вже має бути у файлі
        seen_lines_during_crawl.append(len(out.read_text(encoding="utf-8").splitlines()))
        return httpx.Response(200, html=page_html("Slow", "y"))

    settings = CrawlSettings(seeds=("http://e.com/",), max_pages=5, delay=0.0)
    asyncio.run(run_crawl(settings, out, CrawlStats(), transport=httpx.MockTransport(handler)))
    assert seen_lines_during_crawl == [1]
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_crawl_command_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_test_transport", site_transport())
    out, log = tmp_path / "c.jsonl", tmp_path / "crawl.log"
    result = runner.invoke(
        app,
        ["crawl", "http://e.com/", "--out", str(out), "--log-file", str(log), "--delay", "0",
         "--max-pages", "10", "-c", "3"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Зібрано 3 сторінок" in result.stdout
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if "status=" in ln]
    assert len(lines) == 3  # по рядку на URL (рядок про robots.txt не рахуємо)
    assert all("status=200" in line and "bytes=" in line and "attempts=1" in line for line in lines)


def test_crawl_command_logs_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, html=page_html("H", "x", "/gone"))
        return httpx.Response(404)

    monkeypatch.setattr(cli, "_test_transport", httpx.MockTransport(handler))
    log = tmp_path / "crawl.log"
    result = runner.invoke(
        app, ["crawl", "http://e.com/", "-o", str(tmp_path / "o.jsonl"), "--log-file", str(log),
              "--delay", "0"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "помилок: 1" in result.stdout
    text = log.read_text(encoding="utf-8")
    assert "http://e.com/gone status=404" in text and "error=http_404" in text


def test_crawl_command_rejects_non_http_seed() -> None:
    result = runner.invoke(app, ["crawl", "ftp://x/y"])
    assert result.exit_code == 1
    assert "не http(s)-адреса" in result.output


def test_iter_documents_reads_jsonl_and_directories(tmp_path: Path, tiny_corpus: Path) -> None:
    out = tmp_path / "p.jsonl"
    asyncio.run(
        run_crawl(
            CrawlSettings(seeds=("http://e.com/",), delay=0.0), out, CrawlStats(),
            transport=site_transport(),
        )
    )  # fmt: skip
    texts = list(iter_documents(out))
    assert len(texts) == 3 and any("zebra" in t for t in texts)
    assert [d.path.name for d in iter_jsonl_documents(out)][0].startswith("0000-")
    assert len(list(iter_documents(tiny_corpus))) == 3  # каталог працює як і раніше


def test_page_filename_and_unpack(tmp_path: Path) -> None:
    assert page_filename(7, "http://e.com/p/7") == "0007-p-7.txt"
    assert page_filename(0, "http://e.com/") == "0000-index.txt"
    assert page_filename(3, "http://e.com/a?x=1") == "0003-a-x-1.txt"
    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text(
        json.dumps({"url": "http://e.com/a", "title": "T", "text": "hello world"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "pages"
    dest.mkdir()
    (dest / "0099-stale.txt").write_text("old", encoding="utf-8")
    (dest / "notes.txt").write_text("not mine", encoding="utf-8")
    assert unpack_jsonl(jsonl, dest) == 1
    assert not (dest / "0099-stale.txt").exists()  # наші старі файли прибрано
    assert (dest / "notes.txt").exists()  # чужі лишаються
    assert (dest / "0000-a.txt").read_text(encoding="utf-8") == "T\n\nhello world\n"
    assert (dest / "urls.tsv").read_text(encoding="utf-8") == "0000-a.txt\thttp://e.com/a\n"


def test_index_command_accepts_crawl_jsonl_and_search_finds_page(tmp_path: Path) -> None:
    out = tmp_path / "p.jsonl"
    asyncio.run(
        run_crawl(
            CrawlSettings(seeds=("http://e.com/",), delay=0.0), out, CrawlStats(),
            transport=site_transport(),
        )
    )  # fmt: skip
    index_path = tmp_path / "ix.bin"
    result = runner.invoke(app, ["index", str(out), "-o", str(index_path), "--executor", "serial"])
    assert result.exit_code == 0, result.output
    assert "docs=3" in result.stdout
    found = runner.invoke(app, ["search", str(index_path), "banana", "--json"])
    assert found.exit_code == 0, found.output
    titles = {json.loads(line)["title"] for line in found.stdout.splitlines()}
    assert len(titles) == 2  # alpha і beta


def test_render_stats_shows_counters() -> None:
    stats = CrawlStats(fetched=5, errors=1, retries=2, in_flight=3, queue_depth=9, seen=20)
    console = Console(record=True, width=80)
    console.print(render_stats(stats, 100))
    text = console.export_text()
    assert "5 / 100" in text and "Помилок" in text and "9 (бачили 20)" in text
