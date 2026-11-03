"""crawl(): черга, воркери, ліміти й зупинка. Сайт — словник у пам'яті за MockTransport."""

import asyncio
from collections import Counter
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from findex.crawler.crawl import CrawlStats, Page, crawl, make_user_agent


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def html(*links: str, title: str = "T", body: str = "text") -> str:
    anchors = " ".join(f'<a href="{link}">x</a>' for link in links)
    return f"<html><head><title>{title}</title></head><body>{body} {anchors}</body></html>"


class FakeSite:
    """Сторінки за шляхом + облік запитів, одночасності й моментів старту."""

    def __init__(
        self, pages: dict[str, str], *, latency: float = 0.0, robots: str | None = None
    ) -> None:
        self.pages = pages
        self.latency = latency
        self.robots = robots
        self.hits: Counter[str] = Counter()
        self.agents: set[str] = set()
        self.starts: dict[str, list[float]] = {}
        self._now = 0
        self.max_now = 0
        self.max_now_by_host: dict[str, int] = {}
        self._now_by_host: Counter[str] = Counter()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.netloc.decode(), request.url.raw_path.decode()
        self.agents.add(request.headers["user-agent"])
        if path == "/robots.txt":
            if self.robots is None:
                return httpx.Response(404)
            return httpx.Response(200, text=self.robots)
        self.hits[host + path] += 1
        self.starts.setdefault(host, []).append(asyncio.get_running_loop().time())
        self._now += 1
        self._now_by_host[host] += 1
        self.max_now = max(self.max_now, self._now)
        self.max_now_by_host[host] = max(self.max_now_by_host.get(host, 0), self._now_by_host[host])
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
        finally:
            self._now -= 1
            self._now_by_host[host] -= 1
        body = self.pages.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, html=body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


async def collect(site: FakeSite, seeds: list[str], **kwargs: Any) -> list[Page]:
    kwargs.setdefault("backoff_base", 0.001)
    return [p async for p in crawl(seeds, transport=site.transport(), **kwargs)]


def chain_site(n: int, host_prefix: str = "") -> dict[str, str]:
    """/0 -> /1, /2, ...: кожна сторінка веде на дві наступні (і на себе, і на попередню)."""
    pages: dict[str, str] = {}
    for i in range(n):
        links = [
            f"{host_prefix}/{i}",
            f"{host_prefix}/{(i + 1) % n}",
            f"{host_prefix}/{(i + 2) % n}",
        ]
        pages[f"/{i}"] = html(*links, title=f"Page {i}")
    return pages


def test_crawl_fetches_each_canonical_url_once() -> None:
    site = FakeSite(
        {
            "/": html("/a", "/a/", "/a#frag", "/a?y=1&x=2", "/a?x=2&y=1", "HTTP://E.com/a", "/b"),
            "/a": html("/", "/b"),
            "/b": html("/a", "/"),
        }
    )
    pages = run(collect(site, ["http://e.com/"], max_pages=100))
    assert sorted(p.url for p in pages) == [
        "http://e.com/",
        "http://e.com/a",
        "http://e.com/a?x=2&y=1",
        "http://e.com/b",
    ]
    assert all(count == 1 for count in site.hits.values())


def test_crawl_yields_page_fields() -> None:
    site = FakeSite({"/": html(title="Home", body="Hello there")})
    (page,) = run(collect(site, ["http://e.com/"], max_pages=5))
    assert (page.url, page.title) == ("http://e.com/", "Home")
    assert page.text == "Hello there"
    assert page.fetched_at.endswith("+00:00")


def test_crawl_stops_exactly_at_max_pages_and_leaves_no_tasks() -> None:
    site = FakeSite(chain_site(60), latency=0.005)

    async def go() -> list[Page]:
        pages = await collect(site, ["http://e.com/0"], max_pages=7, concurrency=10)
        assert len(asyncio.all_tasks()) == 1  # лише ця корутина: воркери скасовано й завершено
        return pages

    pages = run(go())
    assert len(pages) == 7
    assert len({p.url for p in pages}) == 7


def test_crawl_ends_when_frontier_is_empty() -> None:
    site = FakeSite(chain_site(5))
    pages = run(collect(site, ["http://e.com/0"], max_pages=1000))
    assert len(pages) == 5


def test_breaking_out_early_cancels_workers() -> None:
    site = FakeSite(chain_site(40), latency=0.01)

    async def go() -> None:
        stats = CrawlStats()
        gen = crawl(["http://e.com/0"], max_pages=100, transport=site.transport(), stats=stats)
        async for _page in gen:
            break
        await gen.aclose()
        assert len(asyncio.all_tasks()) == 1
        assert stats.fetched == 1

    run(go())


def test_crawl_stays_inside_allowed_domains() -> None:
    site = FakeSite(
        {
            "/": html("http://other.org/x", "http://docs.e.com/d", "/in"),
            "/in": html(),
            "/d": html(),
        }
    )
    pages = run(collect(site, ["http://e.com/"], max_pages=50))
    assert sorted(p.url for p in pages) == ["http://e.com/", "http://e.com/in"]
    assert not any("other.org" in key or "docs.e.com" in key for key in site.hits)

    # Піддомен без явного дозволу не береться; з явним allowed_domains — береться.
    site2 = FakeSite({"/": html("http://docs.e.com/d"), "/d": html()})
    assert len(run(collect(site2, ["http://e.com/"], max_pages=50))) == 1
    pages2 = run(collect(site2, ["http://e.com/"], max_pages=50, allowed_domains={"e.com"}))
    assert len(pages2) == 2


def test_crawl_obeys_robots_and_counts_skips() -> None:
    site = FakeSite(
        {"/": html("/private/a", "/ok"), "/ok": html(), "/private/a": html()},
        robots="User-agent: *\nDisallow: /private\n",
    )
    stats = CrawlStats()
    pages = run(collect(site, ["http://e.com/"], max_pages=50, stats=stats))
    assert sorted(p.url for p in pages) == ["http://e.com/", "http://e.com/ok"]
    assert "e.com/private/a" not in site.hits
    assert stats.robots_skipped == 1


def test_crawl_sends_honest_user_agent() -> None:
    site = FakeSite({"/": html()})
    run(
        collect(
            site, ["http://e.com/"], max_pages=1, user_agent=make_user_agent("https://x.org/me")
        )
    )
    assert site.agents == {"findex-crawler/0.6 (+https://x.org/me)"}


def test_global_and_per_host_limits_hold() -> None:
    pages = {**chain_site(30)}
    site = FakeSite(pages, latency=0.01)
    run(collect(site, ["http://e.com/0"], max_pages=30, concurrency=6, per_host=2))
    assert site.max_now_by_host["e.com"] <= 2  # ліміт хоста

    site2 = FakeSite(chain_site(30), latency=0.01)
    run(collect(site2, ["http://e.com/0"], max_pages=30, concurrency=4, per_host=10))
    assert 2 <= site2.max_now <= 4  # глобальний ліміт; і справді було перекриття запитів


def test_concurrency_one_is_sequential() -> None:
    site = FakeSite(chain_site(10), latency=0.005)
    run(collect(site, ["http://e.com/0"], max_pages=10, concurrency=1, per_host=5))
    assert site.max_now == 1


def test_min_delay_between_requests_to_one_host() -> None:
    site = FakeSite(chain_site(8))
    run(collect(site, ["http://e.com/0"], max_pages=8, concurrency=8, per_host=8, delay=0.05))
    starts = sorted(site.starts["e.com"])
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert min(gaps) >= 0.04  # з невеликим допуском на точність таймера


def test_robots_crawl_delay_overrides_smaller_delay() -> None:
    site = FakeSite(chain_site(4), robots="User-agent: *\nCrawl-delay: 1\n")

    async def go() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()
        await collect(site, ["http://e.com/0"], max_pages=3, concurrency=3, per_host=3, delay=0.0)
        return loop.time() - started

    assert run(go()) >= 1.9  # 3 запити з паузою 1 с: перший одразу, ще два через 1 і 2 с


def test_failures_do_not_stop_the_crawl_and_are_counted() -> None:
    attempts: Counter[str] = Counter()

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        attempts[path] += 1
        if path == "/":
            return httpx.Response(200, html=html("/missing", "/flaky", "/busy", "/ok"))
        if path == "/missing":
            return httpx.Response(404)
        if path == "/flaky":  # двічі 503, потім добре
            return httpx.Response(503) if attempts[path] < 3 else httpx.Response(200, html=html())
        if path == "/busy":  # 429 з Retry-After один раз
            if attempts[path] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, html=html())
        return httpx.Response(200, html=html())

    async def go() -> tuple[list[Page], CrawlStats]:
        stats = CrawlStats()
        pages = [
            p
            async for p in crawl(
                ["http://e.com/"],
                max_pages=50,
                transport=httpx.MockTransport(handler),
                stats=stats,
                backoff_base=0.001,
            )
        ]
        return pages, stats

    pages, stats = run(go())
    assert sorted(p.url for p in pages) == [
        "http://e.com/",
        "http://e.com/busy",
        "http://e.com/flaky",
        "http://e.com/ok",
    ]
    assert stats.errors == 1  # лише /missing
    assert stats.retries == 3  # 2 для /flaky + 1 для /busy
    assert attempts["/missing"] == 1  # 404 не повторюється
    assert stats.fetched == 4 and stats.in_flight == 0 and stats.queue_depth == 0


def test_redirect_to_other_domain_is_dropped_and_redirect_target_not_refetched() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "away.org":
            return httpx.Response(200, html=html("/leak"))
        if path == "/":
            return httpx.Response(200, html=html("/go-away", "/old", "/new"))
        if path == "/go-away":
            return httpx.Response(302, headers={"Location": "http://away.org/x"})
        if path == "/old":
            return httpx.Response(301, headers={"Location": "/new"})
        return httpx.Response(200, html=html(title=path))

    async def go() -> list[Page]:
        return [
            p
            async for p in crawl(
                ["http://e.com/"], max_pages=50, transport=httpx.MockTransport(handler)
            )
        ]

    pages = run(go())
    assert sorted(p.url for p in pages) == ["http://e.com/", "http://e.com/new"]  # /new лише раз


def test_relative_links_resolve_against_real_url_with_trailing_slash() -> None:
    # Канонічна адреса «/docs» (без слеша); сервер, як і справжній, відсилає її на «/docs/».
    async def handler(request: httpx.Request) -> httpx.Response:
        match request.url.path:
            case "/docs":
                return httpx.Response(301, headers={"Location": "/docs/"})
            case "/docs/":
                return httpx.Response(200, html=html("a.html"))
            case "/docs/a.html":
                return httpx.Response(200, html=html())
            case _:
                return httpx.Response(404)

    async def go() -> list[Page]:
        return [
            p
            async for p in crawl(
                ["http://e.com/docs/"], max_pages=10, transport=httpx.MockTransport(handler)
            )
        ]

    pages = run(go())
    assert sorted(p.url for p in pages) == ["http://e.com/docs", "http://e.com/docs/a.html"]


def test_invalid_arguments() -> None:
    async def go() -> None:
        async for _ in crawl(["http://e.com/"], max_pages=0):
            pass

    with pytest.raises(ValueError, match="не менше 1"):
        run(go())


def test_empty_seeds_give_empty_crawl() -> None:
    assert run(collect(FakeSite({}), [], max_pages=5)) == []


def test_on_a_site_of_callable_kind_type_checks() -> None:
    _: Callable[[httpx.Request], Any] = FakeSite({})
