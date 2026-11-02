"""Протилежний дослід: робота, обмежена ВВОДОМ-ВИВОДОМ. Тут потоки якраз масштабуються.

Поки потік чекає відповіді з мережі, він відпускає GIL, тож інші потоки в цей час
працюють. Скрипт завантажує список адрес послідовно й на N потоках.

За замовчуванням він сам піднімає ЛОКАЛЬНИЙ сервер зі ШТУЧНОЮ затримкою відповіді
(``--latency``, 50 мс за замовчуванням): це імітація мережі, а не справжній інтернет,
і в README так і треба написати. Щоб мірити справжні адреси, дайте файл зі списком:

    uv run python -m findex.io_benchmark --urls urls.txt

Запуск з імітацією:

    uv run python -m findex.io_benchmark --requests 40
"""

import argparse
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(latency: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            time.sleep(latency)  # «мережа»: сервер відповідає не одразу
            body = b"x" * 20_000
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return  # без рядка в консолі на кожен запит

    return Handler


class Server(ThreadingHTTPServer):
    """Черга підключень за замовчуванням лише 5: при 16 потоках зайві чекали б ~1 с на повтор."""

    request_queue_size = 128


def fetch(url: str) -> int:
    with urllib.request.urlopen(url, timeout=30) as response:
        return len(response.read())


def fetch_all(urls: Sequence[str], workers: int) -> float:
    start = time.perf_counter()
    if workers == 1:
        total = sum(fetch(url) for url in urls)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            total = sum(pool.map(fetch, urls))
    elapsed = time.perf_counter() - start
    if total <= 0:
        raise SystemExit("жодного байта не отримано")
    return elapsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="I/O-бенчмарк: потоки на мережевих запитах.")
    parser.add_argument("--urls", type=Path, help="файл зі списком адрес (по одній у рядку)")
    parser.add_argument(
        "--requests", type=int, default=40, help="скільки запитів до локального сервера"
    )
    parser.add_argument("--latency", type=float, default=0.05, help="штучна затримка сервера, с")
    parser.add_argument("--workers", default="1,2,4,8,16", help="кількості потоків через кому")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    server: Server | None = None
    if args.urls is not None:
        urls = [line.strip() for line in args.urls.read_text(encoding="utf-8").splitlines()]
        urls = [url for url in urls if url]
        source = f"{len(urls)} адрес із {args.urls}"
    else:
        server = Server(("127.0.0.1", 0), make_handler(args.latency))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        urls = [f"http://127.0.0.1:{port}/{i}" for i in range(args.requests)]
        source = (
            f"{args.requests} запитів до ЛОКАЛЬНОГО сервера зі штучною затримкою "
            f"{args.latency * 1000:.0f} мс (імітація мережі)"
        )

    print(f"Python {sys.version.split()[0]}. {source}. Повторів: {args.repeat}.\n")
    print("| Потоків | Час, с (медіана) | Прискорення |")
    print("|---:|---:|---:|")
    base = 0.0
    for workers in sorted({int(w) for w in args.workers.split(",")}):
        median = statistics.median(fetch_all(urls, workers) for _ in range(args.repeat))
        base = base or median
        print(f"| {workers} | {median:.2f} | {base / median:.2f}× |")
    if server is not None:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
