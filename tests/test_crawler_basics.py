"""Канонізація адрес, розбір HTML, fetch і RobotsCache: усе через MockTransport, без мережі."""

import asyncio
import random
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.crawler.fetch import (
    backoff_delay,
    fetch,
    is_retryable_status,
    parse_retry_after,
)
from findex.crawler.parse import parse_html
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, host_of, in_allowed_domains, looks_like_page

type Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def client_for(handler: Handler, **kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


# ---------- канонізація ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM/A", "http://example.com/A"),
        ("http://example.com", "http://example.com/"),
        ("http://example.com/a/", "http://example.com/a"),
        ("http://example.com/a#section", "http://example.com/a"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("http://example.com/a?b=2&a=1", "http://example.com/a?a=1&b=2"),
        ("http://example.com/a?b=2&a=1&a=0", "http://example.com/a?a=0&a=1&b=2"),
        ("http://user:pw@example.com/a", "http://example.com/a"),
        ("http://[::1]:8000/x/", "http://[::1]:8000/x"),
    ],
)
def test_canonicalize(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


@pytest.mark.parametrize("raw", ["mailto:a@b.c", "javascript:void(0)", "ftp://x/y", "", "http://"])
def test_canonicalize_rejects_non_http(raw: str) -> None:
    assert canonicalize(raw) is None


def test_canonicalize_resolves_relative_links() -> None:
    base = "http://example.com/dir/page.html"
    assert canonicalize("other.html", base) == "http://example.com/dir/other.html"
    assert canonicalize("../up.html#x", base) == "http://example.com/up.html"
    assert canonicalize("/root", base) == "http://example.com/root"
    assert canonicalize("//cdn.example.com/a", base) == "http://cdn.example.com/a"


@given(
    st.from_regex(
        r"https?://[a-zA-Z0-9.-]{1,20}(/[a-zA-Z0-9._~-]{0,8}){0,3}(\?[a-z]=[0-9])?", fullmatch=True
    )
)
def test_canonicalize_is_idempotent(url: str) -> None:
    once = canonicalize(url)
    if once is not None:
        assert canonicalize(once) == once


def test_url_helpers() -> None:
    assert host_of("http://example.com:8080/a") == "example.com:8080"
    assert looks_like_page("http://x.com/a.html")
    assert not looks_like_page("http://x.com/photo.JPG")
    assert in_allowed_domains("http://docs.example.com/a", {"example.com"})
    assert in_allowed_domains("http://example.com/a", {"example.com"})
    assert not in_allowed_domains("http://badexample.com/a", {"example.com"})


# ---------- розбір HTML ----------


def test_parse_html_extracts_title_text_links() -> None:
    html = """<html><head><title> Hello  World </title><style>p{}</style></head>
    <body><h1>Head</h1><script>var x = 1;</script>
    <p>Some <b>bold</b> text.</p>
    <a href="/a">A</a> <a href="/a#again">A2</a> <a href="mailto:x@y.z">mail</a>
    <a href="rel.html">R</a></body></html>"""
    page = parse_html(html, "http://example.com/dir/index.html")
    assert page.title == "Hello World"
    assert page.text == "Head Some bold text. A A2 mail R"
    assert page.links == ["http://example.com/a", "http://example.com/dir/rel.html"]


def test_parse_html_separates_block_tags_but_not_inline_ones() -> None:
    page = parse_html("<h1>Page 0</h1><p>unique</p><div>x</div>in<b>side</b>", "http://e.com/")
    assert page.text == "Page 0 unique x inside"


def test_parse_html_respects_base_tag_and_survives_broken_markup() -> None:
    page = parse_html(
        '<base href="http://other.org/x/"><a href="p">P</a><b><i>oops', "http://e.com/"
    )
    assert page.links == ["http://other.org/x/p"]
    assert "oops" in page.text


# ---------- fetch ----------


async def _no_sleep(_seconds: float) -> None:
    return None


def test_fetch_ok_records_status_bytes_and_time() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<p>hi</p>")

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/")
        assert result.ok
        assert (result.status, result.attempts, result.error) == (200, 1, None)
        assert result.body == "<p>hi</p>"
        assert result.bytes == len("<p>hi</p>")
        assert result.elapsed >= 0

    run(go())


def test_fetch_retries_503_with_exponential_backoff_and_jitter() -> None:
    calls = 0
    slept: list[float] = []

    def handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503) if calls < 4 else httpx.Response(200, html="ok")

    async def record(seconds: float) -> None:
        slept.append(seconds)

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(
                client, "http://e.com/", backoff_base=1.0, jitter=0.5, sleep=record,
                rng=random.Random(1),
            )  # fmt: skip
        assert result.ok and result.attempts == 4

    run(go())
    assert len(slept) == 3
    for attempt, wait in enumerate(slept):  # 1, 2, 4 секунди + jitter до 0.5
        assert 2**attempt <= wait <= 2**attempt + 0.5


def test_fetch_gives_up_after_three_retries() -> None:
    calls = 0

    def handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/", sleep=_no_sleep)
        assert not result.ok
        assert (result.status, result.error, result.attempts) == (500, "http_500", 4)

    run(go())
    assert calls == 4  # 1 спроба + 3 повтори


def test_fetch_honors_retry_after_seconds() -> None:
    calls = 0
    slept: list[float] = []

    def handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, html="ok")

    async def record(seconds: float) -> None:
        slept.append(seconds)

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/", sleep=record, jitter=0.25)
        assert result.ok and result.attempts == 2

    run(go())
    assert 7.0 <= slept[0] <= 7.25  # не менше, ніж просив сервер (а власний backoff лише 0.5)


def test_fetch_gives_up_when_retry_after_is_too_long() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3600"})

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/", sleep=_no_sleep, max_wait=30)
        assert (result.error, result.attempts) == ("retry_after_too_long", 1)

    run(go())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_fetch_does_not_retry_other_4xx(status: int) -> None:
    calls = 0

    def handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/", sleep=_no_sleep)
        assert (result.status, result.error, result.attempts) == (status, f"http_{status}", 1)

    run(go())
    assert calls == 1


def test_fetch_times_out_and_retries() -> None:
    calls = 0

    async def handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(5)  # сервер «завис»
        return httpx.Response(200)

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/", timeout=0.02, sleep=_no_sleep)
        assert (result.status, result.error, result.attempts) == (None, "timeout", 4)
        assert result.elapsed < 2  # таймаути спрацювали, а не чекали 5 с

    run(go())
    assert calls == 4


def test_fetch_reports_redirect_loop_without_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"Location": str(request.url)})

    async def go() -> None:
        async with client_for(handler, follow_redirects=True, max_redirects=3) as client:
            result = await fetch(client, "http://e.com/loop", sleep=_no_sleep)
        assert (result.status, result.error, result.attempts) == (None, "redirect_loop", 1)

    run(go())
    assert calls == 4  # перший запит + 3 редіректи, далі httpx здався


def test_fetch_connection_error_is_recorded_not_raised_and_not_retried() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/", sleep=_no_sleep)
        assert (result.error, result.attempts) == ("ConnectError", 1)

    run(go())


def test_fetch_skips_non_html() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG", headers={"Content-Type": "image/png"})

    async def go() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "http://e.com/a")
        assert (result.status, result.error, result.body) == (200, "not_html", None)
        assert result.bytes == 4

    run(go())


def test_parse_retry_after_forms() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Thu, 01 Jan 2026 12:00:30 GMT", now) == 30.0
    assert parse_retry_after("Thu, 01 Jan 2026 11:00:00 GMT", now) == 0.0  # дата вже минула


def test_backoff_delay_doubles_and_is_capped() -> None:
    assert [backoff_delay(n, base=0.5, cap=3) for n in range(5)] == [0.5, 1.0, 2.0, 3, 3]


def test_retryable_statuses() -> None:
    assert is_retryable_status(429) and is_retryable_status(500) and is_retryable_status(599)
    assert not is_retryable_status(404) and not is_retryable_status(403)


# ---------- robots.txt ----------

ROBOTS = "User-agent: *\nDisallow: /private\nCrawl-delay: 2\n"


def test_robots_disallows_and_reads_crawl_delay() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/robots.txt"
        return httpx.Response(200, text=ROBOTS)

    async def go() -> None:
        async with client_for(handler) as client:
            robots = RobotsCache(client, "findex-crawler/0.6 (+x)")
            assert await robots.allowed("http://e.com/ok")
            assert not await robots.allowed("http://e.com/private/a")
            assert await robots.crawl_delay("http://e.com/ok") == 2.0

    run(go())


def test_robots_cached_once_per_host_even_with_concurrent_callers() -> None:
    hits: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        await asyncio.sleep(0.01)
        return httpx.Response(200, text="User-agent: *\nDisallow:\n")

    async def go() -> None:
        async with client_for(handler) as client:
            robots = RobotsCache(client, "bot")
            urls = [f"http://a.com/{i}" for i in range(10)] + [
                f"http://b.com/{i}" for i in range(5)
            ]
            assert all(await asyncio.gather(*(robots.allowed(u) for u in urls)))
            assert robots.fetches == 2

    run(go())
    assert sorted(hits) == ["a.com", "b.com"]  # по одному запиту на хост


def test_robots_missing_file_allows_everything() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def go() -> None:
        async with client_for(handler) as client:
            robots = RobotsCache(client, "bot")
            assert await robots.allowed("http://e.com/anything")
            assert await robots.crawl_delay("http://e.com/anything") == 0.0

    run(go())


def test_robots_server_error_means_do_not_crawl() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def go() -> None:
        async with client_for(handler) as client:
            robots = RobotsCache(client, "bot", backoff_base=0.001)
            assert not await robots.allowed("http://e.com/anything")

    run(go())
