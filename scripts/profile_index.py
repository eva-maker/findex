"""Побудова індексу в одному потоці, без смуги прогресу: для чистого профілю cProfile.

Запуск: uv run python -m cProfile -o docs/index.prof scripts/profile_index.py data/raw
"""

import sys
from pathlib import Path

from findex.index import build_index


def main() -> None:
    build_index(Path(sys.argv[1]))


main()
