"""Статистика: числа Лаби 1 (зі стрімінгу корпусу) і з готового індексу."""

import os
import subprocess
import sys
from pathlib import Path

from findex.models import Index
from findex.stats import compute_stats, index_stats, run_with_measurement
from findex.stats_greedy import compute_stats_greedy


def test_compute_stats_on_tiny_corpus(tiny_corpus: Path) -> None:
    stats = compute_stats(tiny_corpus)
    assert stats["doc_count"] == 3
    assert stats["token_count"] == 6 + 5 + 9
    assert stats["vocab_size"] == 10
    assert stats["top_50"][0] == ("the", 5)


def test_greedy_and_streaming_versions_agree(tiny_corpus: Path) -> None:
    assert compute_stats_greedy(tiny_corpus) == compute_stats(tiny_corpus)


def test_run_with_measurement_adds_time_and_memory(tiny_corpus: Path) -> None:
    stats = run_with_measurement(tiny_corpus)
    assert stats["elapsed_seconds"] >= 0
    assert stats["peak_memory_mb"] >= 0
    assert stats["doc_count"] == 3


def test_index_stats_matches_corpus_stats(tiny_corpus: Path, tiny_index: Index) -> None:
    from_index = index_stats(tiny_index, top=3)
    from_corpus = compute_stats(tiny_corpus)
    assert from_index["doc_count"] == from_corpus["doc_count"]
    assert from_index["token_count"] == from_corpus["token_count"]
    assert from_index["vocab_size"] == from_corpus["vocab_size"]
    assert from_index["top_terms"] == from_corpus["top_50"][:3]
    assert from_index["has_positions"] is False


def test_python_dash_m_findex_works() -> None:
    """`python -m findex` — те саме, що команда `findex` (через __main__.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "findex", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
        },  # help українською: не залежимо від кодування Windows
        check=False,
    )
    assert result.returncode == 0
    assert "search" in result.stdout
