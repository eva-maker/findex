"""Сайт-дзеркало для бенчмарку: локальний сервер на вільному порту, без виходу в інтернет."""

import http.client
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from findex.mirror_server import Server, page_html, serve


@pytest.fixture
def base_url() -> Iterator[str]:
    server: Server = serve(0, pages=40, latency=0.0, faults=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url: str) -> tuple[int, str, dict[str, str]]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), dict(error.headers)


def test_pages_robots_and_stats(base_url: str) -> None:
    status, body, _ = get(base_url + "/p/3")
    assert status == 200 and "zeta3" in body and 'href="/p/' in body
    assert get(base_url + "/p/999")[0] == 404
    assert "Disallow: /private/" in get(base_url + "/robots.txt")[1]
    stats = json.loads(get(base_url + "/__stats")[1])
    assert stats["requests"] == 2 and stats["max_in_flight"] >= 1  # robots і stats не рахуються
    assert stats["cpu"] >= 0


def test_injected_faults_happen_once(base_url: str) -> None:
    status, _, headers = get(base_url + "/p/17")
    assert (status, headers["Retry-After"]) == (429, "1")
    assert get(base_url + "/p/17")[0] == 200  # другий раз уже нормально
    assert [get(base_url + "/p/23")[0] for _ in range(3)] == [503, 503, 200]


def test_page_html_is_deterministic_and_links_to_private_every_25th() -> None:
    assert page_html(7, 400) == page_html(7, 400)
    assert "/private/secret" in page_html(25, 400) and "/private/secret" not in page_html(26, 400)
    assert "/loop/a" in page_html(5, 400)


def test_keepalive_serves_many_requests_over_one_connection() -> None:
    server = serve(0, pages=10, latency=0.0, faults=False, keepalive=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        for number in (1, 2, 3):
            connection.request("GET", f"/p/{number}")
            response = connection.getresponse()
            assert response.status == 200 and f"zeta{number}" in response.read().decode()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
