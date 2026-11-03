"""Бенчмарк краулера (Лаба 6): той самий seed і max_pages, різна concurrency.

Для кожного запуску піднімається СВІЖИЙ локальний сервер (src/findex/mirror_server.py, окремий
процес: нічого не ділить із краулером, лічильники збоїв починаються з нуля), а краулер
працює в окремому підпроцесі: пік RSS чистий, без накопичення від попередніх запусків.
Кожна точка міряється ``--repeat`` разів (3), у таблицю йде медіана.

    uv run python -m findex.crawl_benchmark                 # вимірювання + таблиця
    uv run python -m findex.crawl_benchmark --render docs/bench_crawl.json

Три групи рядків:
  A. per_host = concurrency, пауза 0: чиста швидкість циклу подій (concurrency 1, 5, 20);
  B. concurrency 20, але per_host 2, пауза 0: що коштує ліміт на хост;
  C. concurrency 20, per_host 2, пауза 0.1 с: «ввічливий» режим.
Сервер має штучну затримку відповіді (імітація мережі) і вмикає збої (--faults).
"""

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from findex.crawler.crawl import CrawlStats
from findex.crawler.runner import CrawlSettings, run_crawl
from findex.parallel_benchmark import cpu_model, peak_rss_mb


class RunResult(TypedDict):
    wall: float
    pages: int
    errors: int
    retries: int
    robots_skipped: int
    rss_mb: float | None
    cpu: float  # CPU-секунд краулера за обхід (process_time)
    server_requests: int
    server_max_in_flight: int
    server_cpu: float  # CPU-секунд сервера за обхід


class Row(TypedDict):
    label: str
    concurrency: int
    per_host: int
    delay: float
    runs: list[RunResult]


@dataclass(frozen=True, slots=True)
class Point:
    label: str
    concurrency: int
    per_host: int
    delay: float


def grid(concurrencies: Sequence[int]) -> list[Point]:
    points = [Point("A", c, c, 0.0) for c in concurrencies]
    top = max(concurrencies)
    points.append(Point("B", top, 2, 0.0))
    points.append(Point("C", top, 2, 0.1))
    return points


# ---------- дитина: один обхід у чистому процесі ----------


def child_main(args: argparse.Namespace) -> int:
    settings = CrawlSettings(
        seeds=(f"http://127.0.0.1:{args.port}/",),
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        per_host=args.per_host,
        delay=args.delay,
        timeout=10.0,
    )
    stats = CrawlStats()
    with tempfile.TemporaryDirectory() as tmp:
        start, cpu_start = time.perf_counter(), time.process_time()
        pages = asyncio.run(run_crawl(settings, Path(tmp) / "pages.jsonl", stats))
        wall = time.perf_counter() - start
        cpu = time.process_time() - cpu_start
    result = {
        "wall": wall,
        "pages": pages,
        "errors": stats.errors,
        "retries": stats.retries,
        "robots_skipped": stats.robots_skipped,
        "rss_mb": peak_rss_mb(),
        "cpu": cpu,
    }
    print(json.dumps(result))
    return 0


# ---------- батько: сервер + дитина ----------


def one_run(
    point: Point, max_pages: int, pages_on_site: int, latency: float, keepalive: bool
) -> RunResult:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "findex.mirror_server",
            "--port",
            "0",
            "--pages",
            str(pages_on_site),
            "--latency",
            str(latency),
            "--faults",
        ],  # fmt: skip
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert server.stdout is not None
        port = int(server.stdout.readline().strip().removeprefix("PORT="))
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "findex.crawl_benchmark",
                "--child",
                "--port",
                str(port),
                "--max-pages",
                str(max_pages),
                "--concurrency",
                str(point.concurrency),
                "--per-host",
                str(point.per_host),
                "--delay",
                str(point.delay),
            ],  # fmt: skip
            capture_output=True,
            text=True,
            check=True,
        )
        child: dict[str, float | int | None] = json.loads(done.stdout.strip().splitlines()[-1])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/__stats", timeout=10) as response:
            server_stats = json.loads(response.read())
    finally:
        server.terminate()
        server.wait(timeout=10)
    return {
        "wall": float(child["wall"] or 0.0),
        "pages": int(child["pages"] or 0),
        "errors": int(child["errors"] or 0),
        "retries": int(child["retries"] or 0),
        "robots_skipped": int(child["robots_skipped"] or 0),
        "rss_mb": child["rss_mb"],
        "cpu": float(child["cpu"] or 0.0),
        "server_cpu": float(server_stats["cpu"]),
        "server_requests": int(server_stats["requests"]),
        "server_max_in_flight": int(server_stats["max_in_flight"]),
    }


def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def render(rows: list[Row], meta: dict[str, str | int | float]) -> str:
    lines = [
        f"Python {meta['python']}, {meta['platform']}, CPU: {meta['cpu']}. "
        f"Сайт: локальне дзеркало на {meta['pages_on_site']} сторінок, штучна затримка "
        f"відповіді {float(meta['latency']) * 1000:.0f} мс, max_pages = {meta['max_pages']}, "
        f"повторів: {meta['repeat']} (у таблиці медіана). Keep-alive на сервері: "
        f"{meta.get('keepalive', 'ні')}.",
        "",
        "| Група | Concurrency | Per-host | Пауза, с | Сторінок | Wall, с | Стор/с | "
        "Помилок | Повторів | Пік RSS, МБ | Макс. одночасних на сервері "
        "| CPU краулера, с | CPU сервера, с |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        runs = row["runs"]
        wall = median([r["wall"] for r in runs])
        pages = int(median([r["pages"] for r in runs]))
        rss_values = [r["rss_mb"] for r in runs if r["rss_mb"] is not None]
        rss = f"{median(rss_values):.0f}" if rss_values else "—"
        lines.append(
            f"| {row['label']} | {row['concurrency']} | {row['per_host']} | {row['delay']:g} "
            f"| {pages} | {wall:.2f} | {pages / wall:.1f} "
            f"| {int(median([r['errors'] for r in runs]))} "
            f"| {int(median([r['retries'] for r in runs]))} | {rss} "
            f"| {int(median([r['server_max_in_flight'] for r in runs]))} "
            f"| {median([r.get('cpu', 0.0) for r in runs]):.2f} "
            f"| {median([r.get('server_cpu', 0.0) for r in runs]):.2f} |"
        )
    return "\n".join(lines)


def parent_main(args: argparse.Namespace) -> int:
    if args.render:
        data = json.loads(Path(args.render).read_text(encoding="utf-8"))
        print(render(data["rows"], data["meta"]))
        return 0

    concurrencies = [int(c) for c in args.concurrency_levels.split(",")]
    rows: list[Row] = []
    for point in grid(concurrencies):
        runs: list[RunResult] = []
        for i in range(args.repeat):
            print(
                f"  {point.label}: concurrency={point.concurrency} per_host={point.per_host} "
                f"delay={point.delay:g}  повтор {i + 1}/{args.repeat}",
                file=sys.stderr,
                flush=True,
            )
            runs.append(
                one_run(point, args.max_pages, args.site_pages, args.latency, args.keepalive)
            )
        rows.append(
            {
                "label": point.label,
                "concurrency": point.concurrency,
                "per_host": point.per_host,
                "delay": point.delay,
                "runs": runs,
            }
        )
    meta: dict[str, str | int | float] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "pages_on_site": args.site_pages,
        "latency": args.latency,
        "max_pages": args.max_pages,
        "repeat": args.repeat,
        "keepalive": "так (HTTP/1.1)"
        if args.keepalive
        else "ні (HTTP/1.0, нове з'єднання на запит)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    print(render(rows, meta))
    print(f"\nСирі дані: {out}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Бенчмарк краулера на локальному дзеркалі.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--concurrency", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--per-host", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--delay", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--concurrency-levels", default="1,5,20")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--site-pages", type=int, default=400)
    parser.add_argument("--latency", type=float, default=0.05)
    parser.add_argument(
        "--keepalive", action="store_true", help="сервер тримає з'єднання (HTTP/1.1)"
    )
    parser.add_argument("--out", default="docs/bench_crawl.json")
    parser.add_argument("--render", help="надрукувати таблицю з готового JSON і вийти")
    args = parser.parse_args(argv)
    return child_main(args) if args.child else parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
