"""Один ввічливий запит: таймаут, повтори з exponential backoff + jitter, Retry-After.

Правила повтору (за завданням):
  * повторюємо лише 429, 5xx і таймаут, щонайбільше ``max_retries`` (3) разів;
  * інші 4xx не повторюємо; зациклені редіректи теж (це не тимчасова помилка);
  * очікування = max(backoff, Retry-After) + jitter, де backoff = min(cap, base * 2**спроба);
  * якщо сервер просить чекати довше за ``max_wait``, здаємося одразу.

``sleep`` і ``rng`` можна підмінити: тести не чекають по-справжньому і перевіряють точні числа.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Final

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES: Final = frozenset({429})  # плюс усі 5xx, див. is_retryable_status
HTML_TYPES: Final = ("text/html", "application/xhtml+xml", "text/plain")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Підсумок одного URL (після всіх повторів): те, що йде в crawl.log і таблицю."""

    url: str
    status: int | None  # None — відповіді не було (таймаут, зациклені редіректи, збій з'єднання)
    final_url: str  # після редіректів
    body: str | None  # текст, якщо відповідь 200 і це HTML/текст
    bytes: int
    elapsed: float  # секунд, разом з усіма повторами й очікуваннями
    attempts: int
    error: str | None = None  # "timeout", "redirect_loop", "http_404", "not_html", ...

    @property
    def ok(self) -> bool:
        return self.error is None and self.body is not None


def is_retryable_status(status: int) -> bool:
    return status in RETRY_STATUSES or 500 <= status < 600


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After буває числом секунд або датою HTTP; неправильне значення -> None."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def backoff_delay(attempt: int, *, base: float, cap: float) -> float:
    """Затримка перед повтором номер ``attempt`` (0, 1, 2...): base, 2*base, 4*base..., не більше cap."""
    return min(cap, base * 2**attempt)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 8.0,
    jitter: float = 0.25,
    max_wait: float = 30.0,
    max_bytes: int = 2_000_000,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> FetchResult:
    """Завантажити ``url``. Не кидає винятків за мережеві збої: вони стають полем ``error``."""
    rand = rng or random.Random()
    started = asyncio.get_running_loop().time()
    attempt = 0
    while True:
        status: int | None = None
        error: str | None = None
        retry_after: float | None = None
        body: str | None = None
        size = 0
        final_url = url
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, timeout=None)  # час рахує asyncio.timeout
            status = response.status_code
            final_url = str(response.url)
            size = len(response.content)
            if status == 200:
                kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if kind and kind not in HTML_TYPES:
                    error = "not_html"
                elif size > max_bytes:
                    error = "too_big"
                else:
                    body = response.text
            else:
                error = f"http_{status}"
                retry_after = parse_retry_after(response.headers.get("retry-after"))
        except TimeoutError:
            error = "timeout"
        except httpx.TooManyRedirects:
            error = "redirect_loop"
        except httpx.HTTPError as exc:  # відмова з'єднання, обрив, неправильна відповідь
            error = f"{type(exc).__name__}"

        retryable = error == "timeout" or (status is not None and is_retryable_status(status))
        if error is None or not retryable or attempt >= max_retries:
            elapsed = asyncio.get_running_loop().time() - started
            return FetchResult(url, status, final_url, body, size, elapsed, attempt + 1, error)

        wait = max(backoff_delay(attempt, base=backoff_base, cap=backoff_cap), retry_after or 0.0)
        if wait > max_wait:
            log.warning("%s: сервер просить чекати %.0f с (> %.0f) — здаюся", url, wait, max_wait)
            elapsed = asyncio.get_running_loop().time() - started
            return FetchResult(
                url, status, final_url, None, size, elapsed, attempt + 1, "retry_after_too_long"
            )
        wait += rand.uniform(0.0, jitter)
        log.info(
            "retry %s: спроба %d/%d не вдалась (%s), чекаю %.2f с%s",
            url,
            attempt + 1,
            max_retries + 1,
            error,
            wait,
            f", Retry-After={retry_after:g}" if retry_after is not None else "",
        )
        await sleep(wait)
        attempt += 1
