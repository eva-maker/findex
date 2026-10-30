"""Повільні перевірки на справжньому корпусі з Лаби 1 (20 книг Gutenberg).

Каталог data/raw не потрапляє в git, тому на CI ці тести просто пропускаються.
Локально: `uv run pytest -m slow`; без них: `uv run pytest -m "not slow"`.
"""

from pathlib import Path

import pytest

from findex.index import build_index
from findex.ranking import search

CORPUS = Path("data/raw")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not CORPUS.is_dir(), reason="немає data/raw (корпус не в репозиторії)"),
]


def test_numbers_from_the_readme() -> None:
    index = build_index(CORPUS)
    assert index.num_docs == 20
    assert len(index) == 57_901  # унікальних термінів, як у README Лаби 1/2
    assert sum(p.freq for p in index["the"]) == 84_084


def test_ranked_search_finds_the_war_books() -> None:
    index = build_index(CORPUS)
    titles = [r.title for r in search(index, "war", k=3)]
    assert any("peloponnesian" in t or "lysistrata" in t for t in titles)
