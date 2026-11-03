"""robots.txt: окремо для кожного хоста, з кешем.

Поведінка:
  * 2xx — розбираємо правила (urllib.robotparser);
  * 4xx (немає файла) — дозволено все;
  * 5xx, таймаут, збій мережі — ВОЛІМО НЕ ПОЛІЗТИ: вважаємо, що заборонено все (так робить і Google);
  * ``Crawl-delay`` віддаємо краулеру як мінімальну паузу між запитами до хоста.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import fetch
from findex.crawler.urls import host_of

log = logging.getLogger(__name__)

type SlotFactory = Callable[[str], AbstractAsyncContextManager[None]]


@dataclass(slots=True)
class _Rules:
    parser: RobotFileParser | None  # None — доступу немає, заборонено все
    allow_all: bool = False


class RobotsCache:
    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        *,
        slot: SlotFactory | None = None,
        timeout: float = 10.0,
        backoff_base: float = 0.5,
    ) -> None:
        self._client = client
        self._agent = user_agent
        self._slot = slot
        self._timeout = timeout
        self._backoff_base = backoff_base
        self._rules: dict[str, _Rules] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.fetches = 0  # скільки разів справді завантажували robots.txt (для тестів і звіту)

    async def _load(self, url: str) -> _Rules:
        host = host_of(url)
        robots_url = f"{url.split('://', 1)[0]}://{host}/robots.txt"
        slot = self._slot(host) if self._slot is not None else nullcontext()
        async with slot:
            result = await fetch(
                self._client,
                robots_url,
                timeout=self._timeout,
                max_retries=1,
                backoff_base=self._backoff_base,
            )
        self.fetches += 1
        if result.ok and result.body is not None:
            parser = RobotFileParser()
            parser.parse(result.body.splitlines())
            log.info("robots %s: правила завантажено", host)
            return _Rules(parser)
        if result.status is not None and 400 <= result.status < 500:
            log.info("robots %s: файла немає (HTTP %d) — дозволено все", host, result.status)
            return _Rules(None, allow_all=True)
        log.warning("robots %s: недоступний (%s) — вважаю, що заборонено все", host, result.error)
        return _Rules(None)

    async def _rules_for(self, url: str) -> _Rules:
        host = host_of(url)
        cached = self._rules.get(host)
        if cached is not None:
            return cached
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:  # решта задач цього хоста чекають, а не качають robots.txt вдруге
            cached = self._rules.get(host)
            if cached is None:
                cached = await self._load(url)
                self._rules[host] = cached
            return cached

    async def allowed(self, url: str) -> bool:
        rules = await self._rules_for(url)
        if rules.allow_all:
            return True
        return rules.parser is not None and rules.parser.can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float:
        """Crawl-delay для нашого User-Agent, секунд; 0, якщо не вказано."""
        rules = await self._rules_for(url)
        if rules.parser is None:
            return 0.0
        delay = rules.parser.crawl_delay(self._agent)
        return float(delay) if delay is not None else 0.0
