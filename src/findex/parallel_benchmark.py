"""Бенчмарк Лаби 5: скільки дає паралельна індексація.

Запуск (з кореня репозиторію):

    uv run python -m findex.parallel_benchmark data/raw --tag gil

Скрипт сам проганяє сітку «executor x кількість воркерів», кожну клітинку по
``--repeat`` разів (за замовчуванням 3), і друкує таблицю з МЕДІАНАМИ у форматі
Markdown, готову для README. Результат також пишеться у JSON (``--out``): з нього
``scripts/plot_speedup.py`` малює графік.

Кожен вимір іде в окремому свіжому процесі Python (``--child``): так пік пам'яті
однієї клітинки не псує сусідню, а імпорти й кеші попереднього запуску не
допомагають наступному. Час файлового кешу ОС усереднюється медіаною, а перед
сіткою корпус один раз прочитується «вхолосту».

Колонки:
  wall      — скільки минуло за годинником від плану шматків до готового індексу;
  CPU       — сумарний процесорний час усіх потоків і усіх процесів-воркерів;
  RSS       — пік пам'яті БАТЬКІВСЬКОГО процесу (воркери-процеси не враховані);
  merge     — час послідовного склеювання часткових індексів;
  speedup   — wall послідовного запуску / wall цього запуску;
  f         — «послідовна частка» за Карпом–Флаттом: яку частку роботи довелося б
              вважати непаралельною, щоб за законом Амдала вийшло ТАКЕ прискорення.
              f = (1/S - 1/N) / (1 - 1/N), де S — прискорення, N — воркерів.
              Стеля прискорення дорівнює 1/f. Для S < 1 (повільніше за цикл) не рахується.
"""

import argparse
import json
import os
import pickle
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TypedDict

from findex.corpus import list_document_paths
from findex.index import (
    EXECUTOR_NAMES,
    ExecutorName,
    build_partials,
    merge,
    plan_chunks,
    worker_cpu_seconds,
)

MB = 1024 * 1024


class Run(TypedDict):
    """Один запуск (одна клітинка, одне повторення)."""

    executor: str
    workers: int
    chunks: int
    wall: float
    build_wall: float
    merge_wall: float
    cpu: float
    worker_cpu: float
    rss_mb: float | None
    pickle_bytes: int
    pickle_s: float
    docs: int
    terms: int
    tokens: int
    the_freq: int
    gil: bool | None


class Row(TypedDict):
    """Клітинка таблиці: медіани по повтореннях."""

    executor: str
    workers: int
    chunks: int
    wall: float
    cpu: float
    rss_mb: float | None
    merge_wall: float
    pickle_bytes: int
    pickle_s: float
    speedup: float
    serial_fraction: float | None
    runs: int


class Meta(TypedDict):
    tag: str
    date: str
    cpu_model: str
    logical_cores: int
    python: str
    gil: bool | None
    positions: bool
    repeat: int
    docs: int
    serial_wall: float
    merge_share: float


# --- пам'ять -----------------------------------------------------------------------------

if sys.platform == "win32":

    def peak_rss_mb() -> float | None:
        """Пік робочого набору (Peak Working Set) цього процесу в МБ; на збій — None."""
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS з psapi.h
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            if not ok:
                return None
            return counters.PeakWorkingSetSize / MB
        except (OSError, AttributeError) as error:
            sys.stderr.write(f"RSS на Windows не отримано: {error}\n")
            return None

else:

    def peak_rss_mb() -> float | None:
        """Пік resident set size цього процесу в МБ (ru_maxrss: Linux — КБ, macOS — байти)."""
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / MB if sys.platform == "darwin" else peak / 1024


def gil_enabled() -> bool | None:
    """sys._is_gil_enabled() (є з Python 3.13); None, якщо інтерпретатор його не має."""
    check = getattr(sys, "_is_gil_enabled", None)
    return bool(check()) if callable(check) else None


def cpu_model() -> str:
    """Назва процесора для README (як вдалося дізнатися без сторонніх пакетів)."""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            with key:
                value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
            return " ".join(str(value).split())
        if sys.platform == "darwin":
            done = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return done.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError) as error:
        sys.stderr.write(f"назву процесора не визначено: {error}\n")
    return platform.processor() or "невідомий процесор"


# --- один вимір (дочірній процес) -------------------------------------------------------------


def measure(
    corpus: Path, executor: ExecutorName, workers: int, chunks: int | None, positions: bool
) -> Run:
    paths = list_document_paths(corpus)
    wanted = chunks if chunks is not None else (1 if executor == "serial" else workers)

    cpu_start = time.process_time()
    t0 = time.perf_counter()
    plan = plan_chunks(paths, wanted)
    parts = build_partials(plan, positions=positions, workers=workers, executor=executor)
    t1 = time.perf_counter()
    cpu_built = time.process_time() - cpu_start

    # Ціна передачі даних між процесами: скільки байтів їде з воркера в батька і скільки
    # часу займає dumps+loads цих самих даних. Міряємо ПОЗА виміром wall/CPU, і лише для
    # процесів (для потоків і циклу дані нікуди не передаються).
    pickle_bytes = 0
    pickle_s = 0.0
    if executor == "processes":
        p0 = time.perf_counter()
        for part in parts:
            blob = pickle.dumps(part, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_bytes += len(blob)
            pickle.loads(blob)
        pickle_s = time.perf_counter() - p0

    worker_cpu = worker_cpu_seconds(parts)
    cpu_merge_start = time.process_time()
    t2 = time.perf_counter()
    index = merge(parts)
    t3 = time.perf_counter()
    cpu_merged = time.process_time() - cpu_merge_start

    return Run(
        executor=executor,
        workers=workers,
        chunks=len(plan),
        wall=(t1 - t0) + (t3 - t2),
        build_wall=t1 - t0,
        merge_wall=t3 - t2,
        cpu=cpu_built + cpu_merged + worker_cpu,
        worker_cpu=worker_cpu,
        rss_mb=peak_rss_mb(),
        pickle_bytes=pickle_bytes,
        pickle_s=pickle_s,
        docs=index.num_docs,
        terms=len(index),
        tokens=sum(m.length for m in index.docs()),
        the_freq=sum(p.freq for p in index.get("the", [])),
        gil=gil_enabled(),
    )


# --- сітка (батьківський процес) --------------------------------------------------------------


def run_child(
    corpus: Path, executor: str, workers: int, chunks: int | None, positions: bool
) -> Run:
    command = [
        sys.executable,
        "-m",
        "findex.parallel_benchmark",
        "--child",
        str(corpus),
        "--executor",
        executor,
        "--workers",
        str(workers),
    ]
    if chunks is not None:
        command += ["--chunks", str(chunks)]
    if positions:
        command.append("--positions")
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    sys.stderr.write(done.stderr)  # зазвичай порожньо; тут видно, чому, скажімо, RSS = н/д
    if done.returncode != 0:
        raise SystemExit(f"вимір {executor} x {workers} упав (код {done.returncode})")
    run: Run = json.loads(done.stdout.strip().splitlines()[-1])
    return run


def median_or_none(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def build_grid(workers: Sequence[int], executors: Sequence[str]) -> list[tuple[str, int]]:
    grid: list[tuple[str, int]] = []
    if "serial" in executors:
        grid.append(("serial", 1))
    for executor in executors:
        if executor != "serial":
            grid += [(executor, w) for w in workers]
    return grid


def summarize(
    cells: dict[tuple[str, int], list[Run]],
) -> tuple[list[Row], float, float]:
    """Медіани по повтореннях, прискорення й теоретична стеля за Амдалем."""
    serial_runs = cells.get(("serial", 1))
    base_runs = serial_runs or cells[min(cells, key=lambda key: key[1])]
    serial_wall = statistics.median(r["wall"] for r in base_runs)

    # Послідовна частка, яку можна виміряти напряму: merge. Беремо його з процесів
    # з найбільшою кількістю воркерів (там шматків найбільше, тож merge найдорожчий).
    process_cells = [key for key in cells if key[0] == "processes"]
    merge_source = max(process_cells, key=lambda key: key[1]) if process_cells else None
    merge_wall = (
        statistics.median(r["merge_wall"] for r in cells[merge_source]) if merge_source else 0.0
    )
    merge_share = merge_wall / serial_wall if serial_wall else 0.0

    rows: list[Row] = []
    for (executor, workers), runs in cells.items():
        wall = statistics.median(r["wall"] for r in runs)
        speedup = serial_wall / wall if wall else 0.0
        fraction = None
        if executor != "serial" and workers >= 2 and speedup > 0:
            candidate = (1 / speedup - 1 / workers) / (1 - 1 / workers)
            fraction = candidate if 0 <= candidate <= 1 else None
        rows.append(
            Row(
                executor=executor,
                workers=workers,
                chunks=runs[0]["chunks"],
                wall=wall,
                cpu=statistics.median(r["cpu"] for r in runs),
                rss_mb=median_or_none([r["rss_mb"] for r in runs]),
                merge_wall=statistics.median(r["merge_wall"] for r in runs),
                pickle_bytes=int(statistics.median(r["pickle_bytes"] for r in runs)),
                pickle_s=statistics.median(r["pickle_s"] for r in runs),
                speedup=speedup,
                serial_fraction=fraction,
                runs=len(runs),
            )
        )
    return rows, serial_wall, merge_share


def render_markdown(meta: Meta, rows: Sequence[Row]) -> str:
    gil = {True: "GIL увімкнено", False: "GIL вимкнено", None: "GIL: невідомо"}[meta["gil"]]
    lines = [
        f"Машина: {meta['cpu_model']}, логічних ядер: {meta['logical_cores']}. "
        f"Python {meta['python']} ({gil}). Корпус: {meta['docs']} книг, "
        f"{'з позиціями' if meta['positions'] else 'без позицій'}. "
        f"Кожна клітинка — медіана з {meta['repeat']} запусків.",
        "",
        "| Executor | Воркерів | Шматків | Wall, с | CPU, с | RSS батька, МБ | Merge, с "
        "| Speedup | f |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rss = f"{row['rss_mb']:.0f}" if row["rss_mb"] is not None else "н/д"
        fraction = f"{row['serial_fraction']:.2f}" if row["serial_fraction"] is not None else "—"
        lines.append(
            f"| {row['executor']} | {row['workers']} | {row['chunks']} | {row['wall']:.2f} "
            f"| {row['cpu']:.2f} | {rss} | {row['merge_wall']:.2f} "
            f"| {row['speedup']:.2f}× | {fraction} |"
        )
    process_rows = [r for r in rows if r["executor"] == "processes" and r["pickle_bytes"]]
    if process_rows:
        biggest = max(process_rows, key=lambda r: r["workers"])
        lines += [
            "",
            f"Передача даних між процесами (processes, {biggest['workers']} воркерів): воркери "
            f"повертають батькові {biggest['pickle_bytes'] / MB:.1f} МБ pickle-даних; "
            f"dumps+loads цих даних в одному процесі ≈ {biggest['pickle_s']:.2f} с.",
        ]
    if meta["merge_share"] > 0:
        lines += [
            "",
            f"Merge у батьківському процесі — {meta['merge_share'] * 100:.1f}% від послідовного "
            f"wall ({meta['serial_wall']:.2f} с). Колонка f — послідовна частка, яку виходить "
            "із виміряного прискорення (за Карпом–Флаттом); стеля прискорення = 1/f.",
        ]
    return "\n".join(lines)


def parent_main(args: argparse.Namespace) -> int:
    corpus: Path = args.corpus
    paths = list_document_paths(corpus)
    if not paths:
        raise SystemExit(f"у {corpus} немає .txt-файлів")
    logical = os.cpu_count() or 1
    workers = sorted({int(w) for w in args.workers.split(",")} | {logical})
    executors = [e for e in args.executors.split(",") if e]
    for executor in executors:
        if executor not in EXECUTOR_NAMES:
            raise SystemExit(f"невідомий executor: {executor}")
    grid = build_grid(workers, executors)

    print(f"Корпус: {len(paths)} файлів. Прогрів файлового кешу...", file=sys.stderr)
    for path in paths:
        path.read_bytes()

    cells: dict[tuple[str, int], list[Run]] = {}
    fingerprint: tuple[int, int, int, int] | None = None
    for executor, count in grid:
        for attempt in range(args.repeat):
            print(f"  {executor} x {count}, запуск {attempt + 1}/{args.repeat}", file=sys.stderr)
            run = run_child(corpus, executor, count, args.chunks, args.positions)
            seen = (run["docs"], run["terms"], run["tokens"], run["the_freq"])
            if fingerprint is None:
                fingerprint = seen
            elif seen != fingerprint:
                raise SystemExit(f"{executor} x {count} дав ІНШИЙ індекс: {seen} != {fingerprint}")
            cells.setdefault((executor, count), []).append(run)

    rows, serial_wall, merge_share = summarize(cells)
    first = next(iter(cells.values()))[0]
    meta = Meta(
        tag=args.tag,
        date=date.today().isoformat(),
        cpu_model=cpu_model(),
        logical_cores=logical,
        python=platform.python_version(),
        gil=first["gil"],
        positions=args.positions,
        repeat=args.repeat,
        docs=len(paths),
        serial_wall=serial_wall,
        merge_share=merge_share,
    )
    print(render_markdown(meta, rows))
    if fingerprint is not None:
        print(
            f"\nПеревірка: у всіх клітинках однаковий індекс "
            f"(документів {fingerprint[0]}, термінів {fingerprint[1]}, токенів {fingerprint[2]}, "
            f"«the» {fingerprint[3]})."
        )

    out: Path = args.out or Path(f"docs/bench_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nРезультати збережено в {out}", file=sys.stderr)
    return 0


def render_main(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta: Meta = data["meta"]
    rows: list[Row] = data["rows"]
    print(render_markdown(meta, rows))
    return 0


def child_main(args: argparse.Namespace) -> int:
    run = measure(args.corpus, args.executor, int(args.workers), args.chunks, args.positions)
    print(json.dumps(run))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Бенчмарк паралельної індексації (Лаба 5).")
    parser.add_argument("corpus", type=Path, nargs="?", help="каталог з .txt (напр. data/raw)")
    parser.add_argument("--tag", default="gil", help="мітка запуску в назві файлу і на графіку")
    parser.add_argument(
        "--out", type=Path, help="куди писати JSON (за замовч. docs/bench_<tag>.json)"
    )
    parser.add_argument("--repeat", type=int, default=3, help="скільки разів міряти клітинку")
    parser.add_argument("--workers", default="1,2,4,8", help="кількості воркерів через кому")
    parser.add_argument("--executors", default="serial,threads,processes")
    parser.add_argument("--chunks", type=int, help="кількість шматків (за замовч. = воркерів)")
    parser.add_argument("--positions", action="store_true", help="будувати індекс з позиціями")
    parser.add_argument(
        "--render", type=Path, metavar="JSON", help="лише надрукувати таблицю з JSON"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--executor", default="serial", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.render is not None:
        return render_main(args.render)
    if args.corpus is None:
        parser.error("потрібен каталог з корпусом")
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
