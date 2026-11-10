"""Скільки часу токенізації книжки йде в нативному коді (regex), а скільки в Python.

Запуск: uv run python scripts/split_python_native.py data/raw
Береться найбільша книжка каталогу; кожен варіант міряється 3 рази, береться мінімум.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path

from findex.corpus import read_document
from findex.tokens import _TOKEN_RE, normalize_text, token_spans  # pyright: ignore


def best_of(fn: Callable[[], object], repeat: int = 3) -> float:
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    paths = sorted(Path(sys.argv[1]).glob("*.txt"), key=lambda p: p.stat().st_size)
    path = paths[-1]
    text = normalize_text(read_document(path))
    n = len(_TOKEN_RE.findall(text))
    native = best_of(lambda: _TOKEN_RE.findall(text))
    finditer = best_of(lambda: sum(1 for _ in _TOKEN_RE.finditer(text)))
    full = best_of(lambda: sum(1 for _ in token_spans(text)))
    print(f"{path.name}: {len(text):_} символів, {n:_} токенів")
    print(f"findall (майже весь час у C):             {native * 1000:8.1f} мс")
    print(f"finditer + цикл Python:                   {finditer * 1000:8.1f} мс")
    print(f"token_spans (finditer + group/lower/...): {full * 1000:8.1f} мс")
    print(f"частка нативного сканування: {native / full:.0%}, Python: {1 - native / full:.0%}")


main()
