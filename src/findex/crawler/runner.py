"""Запуск краулера «до файлу»: стрімінговий запис JSON-рядків, без накопичення сторінок у пам'яті."""

import asyncio
import json
from collections.abc import Collection
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from findex.crawler.crawl import CrawlStats, Page, crawl


@dataclass(frozen=True, slots=True)
class CrawlSettings:
    seeds: tuple[str, ...]
    max_pages: int = 500
    concurrency: int = 10
    per_host: int = 2
    allowed_domains: Collection[str] | None = None
    delay: float = 0.25
    timeout: float = 10.0
    backoff_base: float = 0.5
    user_agent: str | None = None


def page_to_line(page: Page) -> str:
    """Один рядок JSONL. ensure_ascii=False: кирилиця лишається читабельною у файлі."""
    return json.dumps(asdict(page), ensure_ascii=False)


async def run_crawl(
    settings: CrawlSettings,
    out_path: Path,
    stats: CrawlStats,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Обійти сайт і писати кожну сторінку у ``out_path`` одразу, як вона надійшла.

    Запис іде через ``asyncio.to_thread``: файловий ввід-вивід блокуючий, і цикл подій не
    повинен його чекати. Файл скидається на диск після кожного рядка, тож ``tail -f`` бачить
    сторінки під час обходу, а не в кінці. Повертає кількість записаних сторінок.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as file:

        def write(line: str) -> None:
            file.write(line + "\n")
            file.flush()

        async for page in crawl(
            settings.seeds,
            max_pages=settings.max_pages,
            concurrency=settings.concurrency,
            per_host=settings.per_host,
            allowed_domains=settings.allowed_domains,
            delay=settings.delay,
            timeout=settings.timeout,
            backoff_base=settings.backoff_base,
            user_agent=settings.user_agent,
            stats=stats,
            transport=transport,
        ):
            await asyncio.to_thread(write, page_to_line(page))
            written += 1
    return written
