"""Збереження/завантаження: round-trip у двох форматах і поведінка при поламаних файлах."""

import pickle
from pathlib import Path

import pytest

from findex.models import Index
from findex.serialize import IndexLoadError, load_index, save_index


def snapshot(
    index: Index,
) -> tuple[dict[str, list[tuple[int, int, tuple[int, ...]]]], list[object]]:
    """Index порівнюється за ідентичністю (потрібно для lru_cache), тому
    для порівняння «чи той самий вміст» робимо звичайний знімок."""
    postings = {t: [(p.doc_id, p.freq, p.positions) for p in pl] for t, pl in index.items()}
    return postings, list(index.docs())


@pytest.mark.parametrize("suffix", [".bin", ".json"])
def test_roundtrip_without_positions(tiny_index: Index, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"ix{suffix}"
    save_index(tiny_index, path)
    assert snapshot(load_index(path)) == snapshot(tiny_index)


@pytest.mark.parametrize("suffix", [".bin", ".json"])
def test_roundtrip_keeps_positions(tiny_index_pos: Index, tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"ix{suffix}"
    save_index(tiny_index_pos, path)
    loaded = load_index(path)
    assert loaded.has_positions is True
    assert snapshot(loaded) == snapshot(tiny_index_pos)


def test_missing_file_is_index_load_error(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError, match="не знайдено"):
        load_index(tmp_path / "nope.bin")


@pytest.mark.parametrize("suffix", [".bin", ".json"])
def test_garbage_file_is_index_load_error(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(b"this is definitely not an index")
    with pytest.raises(IndexLoadError, match="пошкоджений"):
        load_index(path)


def test_pickle_of_something_else_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.bin"
    path.write_bytes(pickle.dumps({"not": "an index"}))
    with pytest.raises(IndexLoadError):
        load_index(path)


def test_truncated_pickle_is_rejected(tiny_index: Index, tmp_path: Path) -> None:
    path = tmp_path / "ix.bin"
    save_index(tiny_index, path)
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(IndexLoadError):
        load_index(path)
