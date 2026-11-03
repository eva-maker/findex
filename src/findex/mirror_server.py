"""Локальний «сайт» для краулера: свій, тож дозвіл на обхід очевидний (це мій власний сервер).

Це штучне дзеркало з N сторінок із посиланнями одна на одну й затримкою відповіді
(``--latency``, імітація мережі). Окремий процес, щоб сервер не ділив GIL із краулером.

    uv run python -m findex.mirror_server --port 8765 --pages 400 --latency 0.05 --faults

З ``--faults`` у сайті з'являються справжні збої, щоб було що обробляти:
  * /p/17 — перший запит отримує 429 із Retry-After: 1;
  * /p/23 — перші два запити отримують 503;
  * /p/5 веде на /loop/a, а /loop/a і /loop/b відсилають одне на одне (зациклений редірект).

/robots.txt забороняє /private/, на який посилаються деякі сторінки: краулер не має туди йти.
/__stats віддає лічильники сервера (скільки запитів, найбільше одночасних, скільки CPU він
спалив), щоб бенчмарк міг перевірити, що краулер справді тримав стільки запитів одночасно,
скільки здається, і чи не впирається вимір у сам сервер.

За замовчуванням сервер говорить HTTP/1.0: після кожної відповіді з'єднання закривається, тож
кожен запит — нове TCP-з'єднання (на Windows це дорого). З ``--keepalive`` він говорить
HTTP/1.1 і тримає з'єднання відкритими, як справжні веб-сервери.
"""

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

VOCAB = (
    "sea ship whale captain harbor anchor storm island river bridge engine garden winter "
    "market letter window candle forest mountain village morning silver copper timber lantern "
    "journey compass meadow orchard harvest granite horizon lantern pottery quarry saddle "
    "thistle voyage wagon yarn zephyr ember falcon glacier heron ivory juniper kettle"
).split()


def page_html(number: int, total: int) -> str:
    rng = random.Random(number)
    topic = VOCAB[number % len(VOCAB)]
    words = rng.choices(VOCAB, k=450)
    paragraphs = [" ".join(words[i : i + 45]) + "." for i in range(0, len(words), 45)]
    targets = {(number + 1) % total, (number * 7 + 1) % total, (number * 13 + 5) % total}
    targets |= {(number * 3 + 2) % total}
    links = [f'<a href="/p/{t}">page {t}</a>' for t in sorted(targets - {number})]
    if number % 25 == 0:
        links.append('<a href="/private/secret">private</a>')  # robots.txt це забороняє
    if number == 5:
        links.append('<a href="/loop/a">loop</a>')
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        f"<html><head><title>Page {number}: {topic}</title></head><body>"
        f"<h1>Page {number}</h1><p>unique term zeta{number}</p>{body}"
        f"<nav>{' '.join(links)}</nav></body></html>"
    )


class State:
    def __init__(self, pages: int, latency: float, faults: bool) -> None:
        self.pages = pages
        self.latency = latency
        self.faults = faults
        self.lock = threading.Lock()
        self.hits: Counter[str] = Counter()
        self.in_flight = 0
        self.max_in_flight = 0
        self.requests = 0
        self.cpu_start = time.process_time()  # щоб не рахувати CPU на запуск інтерпретатора


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = (
        256  # типовий backlog = 5: при 20+ одночасних зʼєднаннях зайві чекали б ~1 с
    )
    state: State

    def handle_error(self, request: object, client_address: object) -> None:
        """Краулер на max_pages обриває запити, що ще йдуть: це не помилка сервера, мовчимо."""
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)  # type: ignore[arg-type]


class Handler(BaseHTTPRequestHandler):
    @property
    def state(self) -> State:
        return cast(Server, self.server).state

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(
        self,
        status: int,
        body: str = "",
        headers: dict[str, str] | None = None,
        ctype: str = "text/html",
    ) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        state = self.state
        path = self.path.split("?", 1)[0]
        if path == "/__stats":
            with state.lock:
                data = {
                    "requests": state.requests,
                    "max_in_flight": state.max_in_flight,
                    "cpu": time.process_time() - state.cpu_start,
                }
            self._send(200, json.dumps(data), ctype="application/json")
            return
        if path == "/robots.txt":
            self._send(200, "User-agent: *\nDisallow: /private/\n", ctype="text/plain")
            return

        with state.lock:
            state.hits[path] += 1
            hit = state.hits[path]
            state.requests += 1
            state.in_flight += 1
            state.max_in_flight = max(state.max_in_flight, state.in_flight)
        try:
            time.sleep(state.latency)  # «мережа»
            self._respond(path, hit, state)
        finally:
            with state.lock:
                state.in_flight -= 1

    def _respond(self, path: str, hit: int, state: State) -> None:
        if path == "/":
            links = " ".join(f'<a href="/p/{i}">page {i}</a>' for i in range(10))
            self._send(200, f"<html><head><title>Mirror</title></head><body>{links}</body></html>")
        elif path.startswith("/p/") and path[3:].isdigit() and int(path[3:]) < state.pages:
            number = int(path[3:])
            if state.faults and number == 17 and hit == 1:
                self._send(429, "slow down", {"Retry-After": "1"}, "text/plain")
            elif state.faults and number == 23 and hit <= 2:
                self._send(503, "try later", ctype="text/plain")
            else:
                self._send(200, page_html(number, state.pages))
        elif path == "/loop/a":
            self._send(302, "", {"Location": "/loop/b"})
        elif path == "/loop/b":
            self._send(302, "", {"Location": "/loop/a"})
        elif path == "/private/secret":
            self._send(200, "<html><head><title>Secret</title></head><body>no</body></html>")
        else:
            self._send(404, "not found", ctype="text/plain")


class KeepAliveHandler(Handler):
    protocol_version = "HTTP/1.1"  # з'єднання лишається відкритим між запитами


def serve(port: int, pages: int, latency: float, faults: bool, keepalive: bool = False) -> Server:
    server = Server(("127.0.0.1", port), KeepAliveHandler if keepalive else Handler)
    server.state = State(pages, latency, faults)
    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Локальний сайт-дзеркало для краулера.")
    parser.add_argument("--port", type=int, default=8765, help="0 — будь-який вільний")
    parser.add_argument("--pages", type=int, default=400)
    parser.add_argument("--latency", type=float, default=0.05, help="штучна затримка відповіді, с")
    parser.add_argument(
        "--faults", action="store_true", help="додати 429, 503 і зациклений редірект"
    )
    parser.add_argument("--keepalive", action="store_true", help="HTTP/1.1: тримати з'єднання")
    args = parser.parse_args(argv)
    server = serve(args.port, args.pages, args.latency, args.faults, args.keepalive)
    print(f"PORT={server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
