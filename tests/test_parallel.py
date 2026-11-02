"""Лаба 5: build_partial, merge, план шматків і три executor-и.

Головна вимога: що б не виконувало збірку (цикл, потоки, процеси) і на скільки
шматків не ділився корпус, індекс виходить той самий, що й послідовний з Лаби 4.
"""

import logging
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app, default_workers
from findex.corpus import Document, list_document_paths, load_document
from findex.index import (
    EXECUTOR_NAMES,
    ExecutorName,
    PartialIndex,
    build_index,
    build_index_parallel,
    build_partial,
    index_documents,
    map_in_executor,
    merge,
    partial_from_documents,
    plan_chunks,
    worker_cpu_seconds,
)
from findex.models import Index
from findex.serialize import load_pickle
from findex.tokens import tokenize

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_findex_logger() -> None:
    """CLI вішає handler на логер «findex»; між тестами прибираємо, щоб не накопичувались."""
    logger = logging.getLogger("findex")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)


type Snapshot = tuple[
    dict[str, list[tuple[int, int, tuple[int, ...]]]],
    list[tuple[int, str, int]],
]


def snapshot(index: Index) -> Snapshot:
    """Весь вміст індексу простими типами (Index порівнюється за ідентичністю, не за вмістом)."""
    postings = {
        term: [(p.doc_id, p.freq, p.positions) for p in index[term]] for term in sorted(index)
    }
    docs = [(m.doc_id, m.path.name, m.length) for m in index.docs()]
    return postings, docs


def naive_oracle(texts: list[str], positions: bool) -> Snapshot:
    """Незалежна проста реалізація: нічого спільного з кодом index.py, крім токенізатора."""
    per_term: dict[str, list[tuple[int, int, tuple[int, ...]]]] = {}
    docs: list[tuple[int, str, int]] = []
    for doc_id, text in enumerate(texts):
        tokens = list(tokenize(text))
        docs.append((doc_id, f"{doc_id}.txt", len(tokens)))
        for term, freq in Counter(tokens).items():
            where = tuple(i for i, t in enumerate(tokens) if t == term) if positions else ()
            per_term.setdefault(term, []).append((doc_id, freq, where))
    return {term: per_term[term] for term in sorted(per_term)}, docs


def tiny_paths(corpus: Path) -> list[Path]:
    return list_document_paths(corpus)


# --- M1: build_partial + merge -------------------------------------------------------------


@pytest.mark.parametrize("positions", [False, True])
def test_merge_of_one_partial_equals_lab4_index(tiny_corpus: Path, positions: bool) -> None:
    paths = tiny_paths(tiny_corpus)
    one_part = merge([build_partial(paths, 0, positions)])
    lab4 = build_index(tiny_corpus, positions=positions)
    assert snapshot(one_part) == snapshot(lab4)


@pytest.mark.parametrize("positions", [False, True])
def test_serial_path_matches_independent_oracle(
    tiny_corpus: Path, tiny_docs: list[Document], positions: bool
) -> None:
    paths = tiny_paths(tiny_corpus)
    index = merge([build_partial(paths, 0, positions)])
    postings, _ = snapshot(index)
    expected, _ = naive_oracle([doc.text for doc in tiny_docs], positions)
    assert postings == expected


def test_partial_uses_the_doc_id_range_it_was_given(tiny_corpus: Path) -> None:
    paths = tiny_paths(tiny_corpus)
    part = build_partial(paths[1:], first_doc_id=10)
    assert sorted(part.doc_meta) == [10, 11]
    assert all(p.doc_id >= 10 for plist in part.postings.values() for p in plist)


def test_build_partial_is_a_plain_module_level_function() -> None:
    """Інакше pickle не відправить її в інший процес (lambda і вкладені функції не їдуть)."""
    assert build_partial.__module__ == "findex.index"
    assert build_partial.__qualname__ == "build_partial"


def test_merge_does_not_care_in_which_order_parts_arrive(tiny_corpus: Path) -> None:
    paths = tiny_paths(tiny_corpus)
    parts = [build_partial(paths[i : i + 1], i) for i in range(3)]
    forward = snapshot(merge(parts))
    backward = snapshot(merge([build_partial(paths[i : i + 1], i) for i in (2, 0, 1)]))
    assert forward == backward


def test_merge_rejects_overlapping_doc_id_ranges(tiny_corpus: Path) -> None:
    paths = tiny_paths(tiny_corpus)
    first = build_partial(paths[:2], 0)
    second = build_partial(paths[1:], 1)  # doc_id 1 уже зайнятий
    with pytest.raises(ValueError, match="перетинаються"):
        merge([first, second])


def test_merge_of_nothing_is_an_empty_index() -> None:
    index = merge([])
    assert len(index) == 0
    assert index.num_docs == 0


def test_postings_stay_sorted_by_doc_id_after_merge(tiny_corpus: Path) -> None:
    paths = tiny_paths(tiny_corpus)
    index = merge([build_partial(paths[i : i + 1], i) for i in range(3)])
    for term in index:
        ids = [p.doc_id for p in index[term]]
        assert ids == sorted(set(ids))


# --- план шматків --------------------------------------------------------------------------


def fake_paths(count: int) -> list[Path]:
    return [Path(f"book{i:02d}.txt") for i in range(count)]


@given(
    sizes=st.lists(st.integers(min_value=0, max_value=10**7), min_size=1, max_size=30),
    parts=st.integers(min_value=1, max_value=40),
)
def test_plan_chunks_covers_every_file_once_in_order(sizes: list[int], parts: int) -> None:
    paths = fake_paths(len(sizes))
    plan = plan_chunks(paths, parts, sizes)
    assert len(plan) == min(parts, len(paths))
    assert all(chunk.paths for chunk in plan)  # порожніх шматків нема
    flat = [path for chunk in plan for path in chunk.paths]
    assert flat == paths
    # діапазони doc_id: перший doc_id шматка = номер його першого файлу
    for chunk in plan:
        assert paths[chunk.first_doc_id] == chunk.paths[0]


def test_plan_chunks_balances_by_size_not_by_count() -> None:
    sizes = [100, 100, 100, 100, 1000, 100, 100, 100]
    plan = plan_chunks(fake_paths(len(sizes)), 2, sizes)
    # Величезна книжка не повинна опинитися у шматку разом із половиною решти.
    assert [len(c.paths) for c in plan] == [5, 3]


def test_plan_chunks_never_makes_more_chunks_than_files() -> None:
    assert len(plan_chunks(fake_paths(3), 8, [5, 5, 5])) == 3


def test_plan_chunks_empty_corpus() -> None:
    assert plan_chunks([], 4) == []


def test_plan_chunks_reads_sizes_from_disk(tiny_corpus: Path) -> None:
    plan = plan_chunks(tiny_paths(tiny_corpus), 2)
    assert sum(len(c.paths) for c in plan) == 3


# --- M2: один код, три бекенди --------------------------------------------------------------


@pytest.mark.parametrize(
    ("executor", "workers"),
    [("serial", 1), ("threads", 1), ("threads", 3), ("processes", 2)],
)
@pytest.mark.parametrize("positions", [False, True])
def test_every_executor_gives_the_serial_index(
    tiny_corpus: Path, executor: ExecutorName, workers: int, positions: bool
) -> None:
    # Процеси стартують повільно (на Windows — секунди), тому для них лише один варіант.
    paths = tiny_paths(tiny_corpus)
    result = build_index_parallel(paths, positions=positions, workers=workers, executor=executor)
    assert snapshot(result) == snapshot(build_index(tiny_corpus, positions=positions))


@pytest.mark.parametrize("chunks", [1, 2, 3, 10])
def test_number_of_chunks_does_not_change_the_index(tiny_corpus: Path, chunks: int) -> None:
    paths = tiny_paths(tiny_corpus)
    result = build_index_parallel(paths, executor="serial", chunks=chunks)
    assert snapshot(result) == snapshot(build_index(tiny_corpus))


def test_workers_must_be_positive(tiny_corpus: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        build_index_parallel(tiny_paths(tiny_corpus), workers=0)


def test_progress_callback_sees_every_file(tiny_corpus: Path) -> None:
    seen: list[int] = []
    build_index_parallel(
        tiny_paths(tiny_corpus),
        executor="threads",
        workers=2,
        on_chunk_done=lambda chunk: seen.append(len(chunk.paths)),
    )
    assert sum(seen) == 3


@given(
    texts=st.lists(st.text(max_size=80), min_size=1, max_size=8),
    cuts=st.sets(st.integers(min_value=1, max_value=7)),
    positions=st.booleans(),
)
def test_any_split_into_ranges_merges_to_the_same_index(
    texts: list[str], cuts: set[int], positions: bool
) -> None:
    """Властивість: як не порізати документи на суцільні шматки, merge дає той самий індекс."""
    docs = [Document(Path(f"{i}.txt"), text) for i, text in enumerate(texts)]
    bounds = [0, *sorted(c for c in cuts if c < len(docs)), len(docs)]
    parts = [
        partial_from_documents(docs[a:b], first_doc_id=a, positions=positions)
        for a, b in zip(bounds, bounds[1:], strict=False)
    ]
    merged = merge(parts)
    whole = index_documents(docs, positions=positions)
    assert snapshot(merged) == snapshot(whole)
    assert snapshot(whole) == naive_oracle(texts, positions)


# --- помилки воркерів: падаємо зі стеком, не зависаємо ---------------------------------------


def with_undecodable_file(corpus: Path) -> list[Path]:
    """Три файли, середній з яких неможливо прочитати як UTF-8: воркер упаде на ньому."""
    bad = corpus / "b.txt"
    bad.write_bytes(b"\xff\xfe not utf-8 \x80")
    return tiny_paths(corpus)


@pytest.mark.parametrize("executor", EXECUTOR_NAMES)
def test_failing_worker_fails_the_whole_build(tiny_corpus: Path, executor: ExecutorName) -> None:
    paths = with_undecodable_file(tiny_corpus)
    with pytest.raises(UnicodeDecodeError):
        build_index_parallel(paths, workers=2, executor=executor, chunks=3)


def test_process_worker_stack_is_chained_to_the_error(tiny_corpus: Path) -> None:
    """Помилка з воркера-процесу приходить разом зі стеком самого воркера."""
    paths = with_undecodable_file(tiny_corpus)
    with pytest.raises(UnicodeDecodeError) as info:
        build_index_parallel(paths, workers=2, executor="processes", chunks=3)
    cause = info.value.__cause__
    assert cause is not None
    assert "build_partial" in str(cause)


def test_missing_file_is_noticed_before_any_worker_starts(tiny_corpus: Path) -> None:
    paths = tiny_paths(tiny_corpus)
    with pytest.raises(FileNotFoundError):
        build_index_parallel([*paths, paths[0].with_name("gone.txt")], executor="processes")


def die_hard(_: int) -> int:
    os._exit(3)  # процес гине без жодного винятку


def test_dead_worker_process_does_not_hang() -> None:
    with pytest.raises(BrokenProcessPool):
        map_in_executor(die_hard, [1, 2], workers=2, executor="processes")


def double(x: int) -> int:
    return x * 2


@pytest.mark.parametrize("executor", EXECUTOR_NAMES)
def test_map_in_executor_keeps_the_order_and_reports_progress(executor: ExecutorName) -> None:
    done: list[int] = []
    result = map_in_executor(
        double, [5, 6, 7, 8], workers=2, executor=executor, on_done=done.append
    )
    assert result == [10, 12, 14, 16]
    assert sorted(done) == [0, 1, 2, 3]


def test_worker_cpu_counts_each_other_process_once() -> None:
    me = os.getpid()
    parts = [
        PartialIndex({}, {}, worker_pid=me, worker_cpu=99.0),  # свій процес не рахуємо
        PartialIndex({}, {}, worker_pid=1001, worker_cpu=1.0),
        PartialIndex({}, {}, worker_pid=1001, worker_cpu=2.5),  # той самий воркер, пізніше
        PartialIndex({}, {}, worker_pid=1002, worker_cpu=0.5),
    ]
    assert worker_cpu_seconds(parts) == pytest.approx(3.0)


# --- CLI: --workers / --executor --------------------------------------------------------------


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_cli_parallel_index_equals_serial_index(
    tiny_corpus: Path, tmp_path: Path, executor: str
) -> None:
    serial_out = tmp_path / "serial.bin"
    parallel_out = tmp_path / "parallel.bin"
    first = runner.invoke(
        app,
        ["index", str(tiny_corpus), "-o", str(serial_out), "--positions", "--executor", "serial"],
    )
    second = runner.invoke(
        app,
        ["index", str(tiny_corpus), "-o", str(parallel_out), "--positions"]
        + ["--executor", executor, "--workers", "2"],
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert snapshot(load_pickle(serial_out)) == snapshot(load_pickle(parallel_out))


def test_cli_rejects_unknown_executor(tiny_corpus: Path) -> None:
    result = runner.invoke(app, ["index", str(tiny_corpus), "--executor", "gpu"])
    assert result.exit_code != 0


def test_cli_rejects_zero_workers(tiny_corpus: Path) -> None:
    result = runner.invoke(app, ["index", str(tiny_corpus), "--workers", "0"])
    assert result.exit_code != 0


def test_cli_serial_ignores_workers_with_a_warning(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.bin"
    result = runner.invoke(
        app, ["index", str(tiny_corpus), "-o", str(out), "--workers", "4", "--executor", "serial"]
    )
    assert result.exit_code == 0, result.output
    assert "ігнорується" in result.stderr


def test_cli_default_is_processes_with_a_sensible_worker_count(
    tiny_corpus: Path, tmp_path: Path
) -> None:
    """Виміри в README показали, що процеси виграють, тому вони типові (воркерів min(4, ядер))."""
    out = tmp_path / "ix.bin"
    result = runner.invoke(app, ["-v", "index", str(tiny_corpus), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "executor=processes" in result.stderr
    assert f"workers={default_workers()}" in result.stderr
    assert 1 <= default_workers() <= 4


def test_python_dash_m_findex_is_safe_for_spawn(tiny_corpus: Path, tmp_path: Path) -> None:
    """Без `if __name__ == "__main__"` у __main__.py кожен воркер запускав би CLI ще раз."""
    out = tmp_path / "ix.bin"
    done = subprocess.run(
        [sys.executable, "-m", "findex", "index", str(tiny_corpus), "-o", str(out)]
        + ["--executor", "processes", "--workers", "2"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert load_pickle(out).num_docs == 3


# --- справжній корпус (повільно) ---------------------------------------------------------------

CORPUS = Path("data/raw")


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.is_dir(), reason="немає data/raw (корпус не в репозиторії)")
def test_real_corpus_processes_equal_serial_and_readme_numbers() -> None:
    paths = list_document_paths(CORPUS)
    serial = build_index_parallel(paths, executor="serial")
    parallel = build_index_parallel(paths, workers=4, executor="processes")
    assert snapshot(parallel) == snapshot(serial)
    assert len(parallel) == 57_901  # термінів, як у README Лаби 1
    assert sum(p.freq for p in parallel["the"]) == 84_084
    assert sum(m.length for m in parallel.docs()) == 1_374_875  # токенів, як у README Лаби 1


def test_load_document_is_what_workers_read(tiny_corpus: Path, tiny_docs: list[Document]) -> None:
    """Воркер читає файл тим самим load_document, що й послідовний шлях."""
    path = tiny_paths(tiny_corpus)[0]
    assert load_document(path).text == tiny_docs[0].text
