"""Гонка даних: GIL захищає інтерпретатор, але не ваш інваріант.

Кілька потоків одночасно рахують токени справжнього тексту різними способами. Скільки
токенів було насправді, відомо заздалегідь (``expected``), тож видно, чи загубилось
щось по дорозі.

    uv run python -m findex.race_demo data/raw
    uv run python -m findex.race_demo data/raw --switch-interval 0.000001

Те саме на збірці без GIL (потрібна PYTHONPATH, бо проєкт вимагає Python 3.14):

    $env:PYTHONPATH = "src"
    uv run --no-project --python 3.13t python -m findex.race_demo data/raw

Варіанти:
  box_plain      — спільне число: box["n"] += 1 (читання, додавання й запис — окремі кроки);
  counter_plain  — спільний Counter: shared[token] += 1 (те саме для кожного терміна);
  counter_lock   — той самий Counter, але оновлення під threading.Lock;
  counter_local  — кожен потік рахує у своєму Counter, після join їх додають (як merge);
  list_append    — спільний список, лише .append(token): одна операція, вона атомарна.
"""

import argparse
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import list_document_paths, load_document
from findex.tokens import tokenize


@dataclass(slots=True)
class Outcome:
    variant: str
    expected: int
    got: list[int]
    seconds: list[float]


def split(tokens: list[str], parts: int) -> list[list[str]]:
    size = -(-len(tokens) // parts)
    return [tokens[i : i + size] for i in range(0, len(tokens), size)]


def run_threads(work: Sequence[Callable[[], None]]) -> float:
    """Запустити всі функції в окремих потоках одночасно (Barrier) і дочекатись."""
    barrier = threading.Barrier(len(work))

    def runner(fn: Callable[[], None]) -> None:
        barrier.wait()
        fn()

    threads = [threading.Thread(target=runner, args=(fn,)) for fn in work]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - start


def box_plain(chunks: list[list[str]]) -> tuple[int, float]:
    box = {"n": 0}

    def make(chunk: list[str]) -> Callable[[], None]:
        def work() -> None:
            for _ in chunk:
                box["n"] += 1

        return work

    seconds = run_threads([make(chunk) for chunk in chunks])
    return box["n"], seconds


def counter_plain(chunks: list[list[str]]) -> tuple[int, float]:
    shared: Counter[str] = Counter()

    def make(chunk: list[str]) -> Callable[[], None]:
        def work() -> None:
            for token in chunk:
                shared[token] += 1

        return work

    seconds = run_threads([make(chunk) for chunk in chunks])
    return shared.total(), seconds


def counter_lock(chunks: list[list[str]]) -> tuple[int, float]:
    shared: Counter[str] = Counter()
    lock = threading.Lock()

    def make(chunk: list[str]) -> Callable[[], None]:
        def work() -> None:
            for token in chunk:
                with lock:
                    shared[token] += 1

        return work

    seconds = run_threads([make(chunk) for chunk in chunks])
    return shared.total(), seconds


def counter_local(chunks: list[list[str]]) -> tuple[int, float]:
    locals_: list[Counter[str]] = [Counter() for _ in chunks]

    def make(chunk: list[str], mine: Counter[str]) -> Callable[[], None]:
        def work() -> None:
            for token in chunk:
                mine[token] += 1

        return work

    start = time.perf_counter()
    run_threads([make(chunk, mine) for chunk, mine in zip(chunks, locals_, strict=True)])
    total: Counter[str] = Counter()
    for mine in locals_:  # «merge»: тут уже один потік
        total.update(mine)
    return total.total(), time.perf_counter() - start


def list_append(chunks: list[list[str]]) -> tuple[int, float]:
    shared: list[str] = []

    def make(chunk: list[str]) -> Callable[[], None]:
        def work() -> None:
            for token in chunk:
                shared.append(token)

        return work

    seconds = run_threads([make(chunk) for chunk in chunks])
    return len(shared), seconds


VARIANTS: dict[str, Callable[[list[list[str]]], tuple[int, float]]] = {
    "box_plain": box_plain,
    "counter_plain": counter_plain,
    "counter_lock": counter_lock,
    "counter_local": counter_local,
    "list_append": list_append,
}


def gil_label() -> str:
    check = getattr(sys, "_is_gil_enabled", None)
    if not callable(check):
        return "невідомо (старий Python)"
    return "увімкнено" if check() else "ВИМКНЕНО (free-threaded)"


def collect_tokens(corpus: Path, books: int) -> list[str]:
    tokens: list[str] = []
    for path in list_document_paths(corpus)[:books]:
        tokens.extend(tokenize(load_document(path).text))
    return tokens


def render(outcomes: list[Outcome], threads: int, repeat: int, tokens: int) -> str:
    lines = [
        f"Python {sys.version.split()[0]}, GIL {gil_label()}, switchinterval "
        f"{sys.getswitchinterval():g} с. Потоків: {threads}, токенів: {tokens:,}, "
        f"повторів: {repeat}.",
        "",
        "| Варіант | Очікувалось | Вийшло (мін…макс) | Втрачено (найгірше) | Час, с (медіана) |",
        "|---|---:|---:|---:|---:|",
    ]
    for outcome in outcomes:
        lowest, highest = min(outcome.got), max(outcome.got)
        lost = outcome.expected - lowest
        share = lost / outcome.expected * 100
        seconds = sorted(outcome.seconds)[len(outcome.seconds) // 2]
        lines.append(
            f"| {outcome.variant} | {outcome.expected:,} | {lowest:,}…{highest:,} "
            f"| {lost:,} ({share:.3f}%) | {seconds:.3f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Демонстрація гонки даних між потоками.")
    parser.add_argument("corpus", type=Path, help="каталог з .txt (напр. data/raw)")
    parser.add_argument("--threads", type=int, default=4, help="скільки потоків")
    parser.add_argument("--books", type=int, default=3, help="скільки перших книжок взяти")
    parser.add_argument("--repeat", type=int, default=5, help="скільки разів повторити")
    parser.add_argument("--switch-interval", type=float, help="sys.setswitchinterval, секунди")
    args = parser.parse_args(argv)

    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
    tokens = collect_tokens(args.corpus, args.books)
    chunks = split(tokens, args.threads)
    expected = len(tokens)

    outcomes: list[Outcome] = []
    for name, variant in VARIANTS.items():
        outcome = Outcome(variant=name, expected=expected, got=[], seconds=[])
        for _ in range(args.repeat):
            got, seconds = variant(chunks)
            outcome.got.append(got)
            outcome.seconds.append(seconds)
        outcomes.append(outcome)
    print(render(outcomes, len(chunks), args.repeat, expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
