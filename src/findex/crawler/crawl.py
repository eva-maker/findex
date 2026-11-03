"""Асинхронний краулер: черга адрес, воркери в TaskGroup, ліміти, ввічливість.

Схема:

    seeds -> frontier (Queue) -> N воркерів -> fetch -> parse -> нові адреси -> frontier
                                                  \\-> out (Queue) -> async-генератор crawl() -> Page

* Один семафор на ВСІ запити (``concurrency``) і окремий на кожен хост (``per_host``). Порядок
  захоплення завжди однаковий (хост -> пауза -> глобальний), тож взаємного блокування нема.
* Між стартами запитів до одного хоста минає щонайменше ``delay`` секунд
  (або Crawl-delay з robots.txt, якщо він більший).
* ``seen`` не дає поставити в чергу одну канонічну адресу двічі.
* Воркерів удвічі більше за ``concurrency``: воркер, що чекає на ліміт хоста, не займає
  «місце» в загальному ліміті, його тримає лише семафор.
* Токенізації й індексації тут нема: вони йдуть поза циклом подій (див. ``findex index``).
"""

import asyncio
import logging
import ssl
import time
from collections.abc import AsyncGenerator, Collection
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.parse import parse_html
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, host_of, in_allowed_domains, looks_like_page

log = logging.getLogger(__name__)
access = logging.getLogger(__name__ + ".access")  # один рядок на кожен URL -> crawl.log

DEFAULT_CONTACT = "https://github.com/eva-maker/findex"


def make_user_agent(contact: str = DEFAULT_CONTACT, version: str = "0.6") -> str:
    """Чесний User-Agent: хто ми і куди писати, якщо краулер заважає."""
    return f"findex-crawler/{version} (+{contact})"


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str  # ISO 8601, UTC


@dataclass(slots=True)
class CrawlStats:
    """Лічильники для живого екрана; краулер оновлює їх, хтось зовні читає."""

    fetched: int = 0  # сторінок віддано споживачу
    errors: int = 0  # URL, що не вдалося завантажити (після всіх повторів)
    retries: int = 0  # скільки зайвих спроб було (429, 5xx, таймаут)
    robots_skipped: int = 0  # заборонено robots.txt
    in_flight: int = 0  # запитів, що чекають відповіді просто зараз
    queue_depth: int = 0  # адрес у черзі
    seen: int = 0  # скільки різних адрес бачили
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def pages_per_second(self) -> float:
        return self.fetched / self.elapsed if self.elapsed > 0 else 0.0


class Limiter:
    """Глобальний семафор + семафор на хост + мінімальна пауза між стартами запитів до хоста."""

    def __init__(self, concurrency: int, per_host: int, delay: float) -> None:
        self._global = asyncio.Semaphore(concurrency)
        self._per_host = per_host
        self._delay = delay
        self._host_sems: dict[str, asyncio.Semaphore] = {}
        self._host_delay: dict[str, float] = {}
        self._next_start: dict[str, float] = {}

    def set_host_delay(self, host: str, delay: float) -> None:
        """Пауза для хоста: більша з налаштованої і з robots.txt (Crawl-delay)."""
        self._host_delay[host] = max(self._delay, delay)

    async def _pause(self, host: str) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        start = max(now, self._next_start.get(host, 0.0))
        self._next_start[host] = start + self._host_delay.get(host, self._delay)
        if start > now:
            await asyncio.sleep(start - now)  # саме asyncio.sleep: цикл у цей час вільний

    @asynccontextmanager
    async def slot(self, host: str) -> AsyncGenerator[None]:
        sem = self._host_sems.setdefault(host, asyncio.Semaphore(self._per_host))
        async with sem:
            await self._pause(host)
            async with self._global:
                yield


def _log_result(result: FetchResult) -> None:
    access.info(
        "%s status=%s bytes=%d ms=%.0f attempts=%d%s",
        result.url,
        result.status if result.status is not None else "-",
        result.bytes,
        result.elapsed * 1000,
        result.attempts,
        f" error={result.error}" if result.error else "",
    )


async def crawl(
    seeds: Collection[str],
    *,
    max_pages: int,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: Collection[str] | None = None,
    delay: float = 0.0,
    timeout: float = 10.0,
    backoff_base: float = 0.5,
    user_agent: str | None = None,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncGenerator[Page]:
    """Обійти сайт від ``seeds`` і віддавати сторінки по одній, щойно вони завантажені.

    Закінчується, коли віддано ``max_pages`` сторінок або черга вичерпана. Якщо споживач
    кинув ітерацію раніше (``break``), воркери скасовуються, а клієнт закривається.
    """
    if max_pages < 1 or concurrency < 1 or per_host < 1:
        raise ValueError("max_pages, concurrency і per_host мають бути не менше 1")
    stats = stats if stats is not None else CrawlStats()
    agent = user_agent or make_user_agent()

    start_urls = [c for seed in seeds if (c := canonicalize(seed)) is not None]
    # Без явного списку лишаємось на хостах початкових адрес (піддомени — лише якщо дозволено явно).
    subdomains = bool(allowed_domains)
    domains = (
        set(allowed_domains)
        if allowed_domains
        else {urlsplit(u).hostname or "" for u in start_urls}
    )

    frontier: asyncio.Queue[str] = asyncio.Queue()
    out: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    for url in start_urls:
        if url not in seen:
            seen.add(url)
            frontier.put_nowait(url)
    stats.seen = len(seen)
    stats.queue_depth = frontier.qsize()

    limiter = Limiter(concurrency, per_host, delay)

    # Створення клієнта дороге: читаються сертифікати для TLS, перевіряються змінні середовища
    # з проксі. На моєму ПК TLS-контекст блокував цикл на ~0,76 с, а решта конструктора ще
    # ~0,1 с (обидва випадки спіймав asyncio debug). Тож клієнта будуємо у потоці.
    def make_client() -> httpx.AsyncClient:
        verify: ssl.SSLContext | bool = (
            True if transport is not None else httpx.create_ssl_context()
        )
        return httpx.AsyncClient(
            headers={"User-Agent": agent},
            verify=verify,
            follow_redirects=True,
            max_redirects=5,
            transport=transport,
            timeout=None,  # таймаути рахує asyncio.timeout у fetch
        )

    built = time.perf_counter()
    client = await asyncio.to_thread(make_client)
    log.info(
        "HTTP-клієнт створено за %.0f мс (у потоці, цикл вільний)",
        (time.perf_counter() - built) * 1000,
    )

    async with client:
        robots = RobotsCache(
            client, agent, slot=limiter.slot, timeout=timeout, backoff_base=backoff_base
        )

        async def process(url: str) -> None:
            if not await robots.allowed(url):
                stats.robots_skipped += 1
                access.info("%s SKIP robots.txt", url)
                return
            host = host_of(url)
            limiter.set_host_delay(host, await robots.crawl_delay(url))
            async with limiter.slot(host):
                stats.in_flight += 1
                try:
                    result = await fetch(client, url, timeout=timeout, backoff_base=backoff_base)
                finally:
                    stats.in_flight -= 1
            stats.retries += result.attempts - 1
            _log_result(result)
            if not result.ok or result.body is None:
                stats.errors += 1
                return

            final = canonicalize(result.final_url) or url
            if final != url:  # редірект: запам'ятовуємо й кінцеву адресу
                if final in seen or not in_allowed_domains(final, domains, subdomains=subdomains):
                    return
                seen.add(final)
            # Відносні посилання розв'язуємо від справжньої адреси (зі слешем), а не від канонічної:
            # «/docs/» + «a.html» = «/docs/a.html», а «/docs» + «a.html» = «/a.html».
            parsed = parse_html(result.body, result.final_url)
            for link in parsed.links:
                if (
                    link not in seen
                    and looks_like_page(link)
                    and in_allowed_domains(link, domains, subdomains=subdomains)
                ):
                    seen.add(link)
                    frontier.put_nowait(link)
            stats.seen = len(seen)
            stats.queue_depth = frontier.qsize()
            fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
            out.put_nowait(Page(final, parsed.title, parsed.text, fetched_at))

        async def worker() -> None:
            while True:
                url = await frontier.get()
                stats.queue_depth = frontier.qsize()
                try:
                    await process(url)
                finally:
                    frontier.task_done()

        async def finish() -> None:
            await frontier.join()  # черга порожня й жодна адреса не в обробці
            out.put_nowait(None)

        async def supervise() -> None:
            """Воркери живуть у TaskGroup: помилка одного скасовує решту й підіймається нагору.

            Сама TaskGroup відкрита не в генераторі, а в окремій задачі: `yield` усередині
            `async with TaskGroup` ламається, коли споживач просто кидає ітерацію (aclose
            кидає GeneratorExit у групу, і та пакує його в ExceptionGroup).
            """
            try:
                async with asyncio.TaskGroup() as group:
                    for _ in range(concurrency * 2):
                        group.create_task(worker())
                    group.create_task(finish())
            finally:
                out.put_nowait(None)  # хай що сталося, розбудити споживача

        runner = asyncio.create_task(supervise())
        try:
            while (page := await out.get()) is not None:
                stats.fetched += 1
                yield page
                if stats.fetched >= max_pages:
                    break
        finally:  # зупинка: на max_pages, на break споживача чи після помилки воркера
            runner.cancel()
            await asyncio.wait([runner])
        if runner.done() and not runner.cancelled() and (error := runner.exception()) is not None:
            raise error
