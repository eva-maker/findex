import time
from pathlib import Path

from pytest_benchmark.fixture import BenchmarkFixture

from findex.index import build_index
from findex.models import Index
from findex.ranking import search
from findex.vectorized import search_vectorized
from findex.web.service import run_search, run_search_baseline

QUERY = "aaa OR aab OR aac"


def test_bench_search_lab3(benchmark: BenchmarkFixture, synthetic_index: Index) -> None:
    benchmark(search, synthetic_index, QUERY)


def test_bench_search_numpy(benchmark: BenchmarkFixture, synthetic_index: Index) -> None:
    benchmark(search_vectorized, synthetic_index, QUERY)


def test_bench_build_index(benchmark: BenchmarkFixture, synthetic_corpus: Path) -> None:
    benchmark(build_index, synthetic_corpus)


def test_bench_page_baseline(benchmark: BenchmarkFixture, synthetic_index: Index) -> None:
    benchmark(run_search_baseline, synthetic_index, None, QUERY, 10, 1, "bm25")


def test_bench_page_numpy(benchmark: BenchmarkFixture, synthetic_index: Index) -> None:
    run_search(synthetic_index, None, QUERY, 10, 1, "bm25")
    benchmark(run_search, synthetic_index, None, QUERY, 10, 1, "bm25")


def test_fast_page_is_not_slower_than_baseline(synthetic_index: Index) -> None:
    def best(fn: object, rounds: int = 5) -> float:
        times: list[float] = []
        for _ in range(rounds):
            started = time.perf_counter()
            assert callable(fn)
            fn()
            times.append(time.perf_counter() - started)
        return min(times)

    run_search(synthetic_index, None, QUERY, 10, 1, "bm25")
    fast = best(lambda: run_search(synthetic_index, None, QUERY, 10, 1, "bm25"))
    slow = best(lambda: run_search_baseline(synthetic_index, None, QUERY, 10, 1, "bm25"))
    assert fast < slow / 2
