"""Багато разів виконує пошук, щоб профайлер встиг набрати зразки.

Запуск: uv run python scripts/profile_search.py index.bin data/raw "war OR money OR gold" 200
Індекс вантажиться один раз, далі йде той самий виклик, що й у веб-сервісі.
"""

import sys
from pathlib import Path

from findex.serialize import load_index
from findex.web.service import run_search


def main() -> None:
    index = load_index(Path(sys.argv[1]))
    corpus = Path(sys.argv[2])
    query = sys.argv[3]
    repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    for _ in range(repeat):
        run_search(index, corpus, query, 10, 1, "bm25")


main()
