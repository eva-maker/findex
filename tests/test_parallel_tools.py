"""Допоміжні скрипти Лаби 5: бенчмарк, демонстрація гонки, I/O-бенчмарк.

Самі виміри не перевіряємо (числа залежать від машини), перевіряємо, що скрипти
збирають таблицю правильно і що в них нічого не ламається на маленькому корпусі.
"""

from pathlib import Path

import pytest

from findex import io_benchmark, parallel_benchmark, race_demo
from findex.index import ExecutorName
from findex.parallel_benchmark import Meta, Run, measure, render_markdown, summarize


def fake_run(executor: str, workers: int, wall: float, merge: float = 0.1) -> Run:
    return Run(
        executor=executor,
        workers=workers,
        chunks=workers,
        wall=wall,
        build_wall=wall - merge,
        merge_wall=merge,
        cpu=wall * workers,
        worker_cpu=0.0,
        rss_mb=100.0,
        pickle_bytes=2_000_000 if executor == "processes" else 0,
        pickle_s=0.2 if executor == "processes" else 0.0,
        docs=20,
        terms=57_901,
        tokens=1_374_875,
        the_freq=84_084,
        gil=True,
    )


def test_summarize_takes_medians_and_computes_speedup() -> None:
    cells = {
        ("serial", 1): [fake_run("serial", 1, 10.0), fake_run("serial", 1, 12.0)],
        ("processes", 2): [fake_run("processes", 2, 5.0), fake_run("processes", 2, 7.0)],
        ("threads", 2): [fake_run("threads", 2, 14.0)],
    }
    rows, serial_wall, merge_share = summarize(cells)
    by_key = {(r["executor"], r["workers"]): r for r in rows}
    assert serial_wall == pytest.approx(11.0)  # медіана 10 і 12
    assert by_key[("processes", 2)]["wall"] == pytest.approx(6.0)
    assert by_key[("processes", 2)]["speedup"] == pytest.approx(11.0 / 6.0)
    assert by_key[("threads", 2)]["speedup"] < 1
    assert by_key[("threads", 2)]["serial_fraction"] is None  # повільніше за цикл: f не рахуємо
    assert 0 < merge_share < 1


def test_karp_flatt_fraction_matches_amdahl() -> None:
    """Якщо послідовна частка 20%, то 4 воркери дають 1/(0.2+0.8/4) = 2.5×; f має вийти 0.2."""
    cells = {
        ("serial", 1): [fake_run("serial", 1, 10.0)],
        ("processes", 4): [fake_run("processes", 4, 4.0)],
    }
    rows, _, _ = summarize(cells)
    row = next(r for r in rows if r["executor"] == "processes")
    assert row["speedup"] == pytest.approx(2.5)
    assert row["serial_fraction"] == pytest.approx(0.2)


def test_render_markdown_has_a_row_per_cell() -> None:
    cells = {
        ("serial", 1): [fake_run("serial", 1, 10.0)],
        ("processes", 2): [fake_run("processes", 2, 6.0)],
    }
    rows, serial_wall, merge_share = summarize(cells)
    meta = Meta(
        tag="t",
        date="2026-11-02",
        cpu_model="Test CPU",
        logical_cores=8,
        python="3.14.0",
        gil=True,
        positions=False,
        repeat=3,
        docs=20,
        serial_wall=serial_wall,
        merge_share=merge_share,
    )
    table = render_markdown(meta, rows)
    assert "| serial | 1 |" in table
    assert "| processes | 2 |" in table
    assert "Test CPU" in table and "GIL увімкнено" in table
    assert "Передача даних між процесами" in table


def test_build_grid_always_starts_with_serial() -> None:
    grid = parallel_benchmark.build_grid([1, 2], ["serial", "threads", "processes"])
    assert grid == [
        ("serial", 1),
        ("threads", 1),
        ("threads", 2),
        ("processes", 1),
        ("processes", 2),
    ]


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_measure_builds_the_same_index_everywhere(
    tiny_corpus: Path, executor: ExecutorName
) -> None:
    run = measure(tiny_corpus, executor, 2, None, positions=False)
    assert (run["docs"], run["tokens"]) == (3, 20)  # 6 + 4 + 10 токенів у трьох «книжках»
    # CPU time на Windows міряється з кроком ~15 мс, тож на крихітному корпусі може бути 0.0.
    assert run["wall"] > 0 and run["cpu"] >= 0


def test_race_demo_split_keeps_every_token() -> None:
    tokens = [str(i) for i in range(10)]
    chunks = race_demo.split(tokens, 3)
    assert [t for chunk in chunks for t in chunk] == tokens


@pytest.mark.parametrize("name", ["counter_lock", "counter_local", "list_append"])
def test_safe_race_variants_never_lose_updates(name: str) -> None:
    tokens = [str(i % 7) for i in range(5000)]
    chunks = race_demo.split(tokens, 4)
    got, seconds = race_demo.VARIANTS[name](chunks)
    assert got == len(tokens)
    assert seconds >= 0


def test_io_benchmark_runs_against_its_local_server(capsys: pytest.CaptureFixture[str]) -> None:
    code = io_benchmark.main(
        ["--requests", "6", "--latency", "0.01", "--workers", "1,3", "--repeat", "1"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "ЛОКАЛЬНОГО" in out
    assert "| 3 |" in out
