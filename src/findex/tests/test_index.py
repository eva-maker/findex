from pathlib import Path

import pytest

from findex.models import DocMeta, Index, Posting
from findex.serialize import open_index, save_index


@pytest.fixture
def index() -> Index:
    postings = {
        "cat": [Posting(0, 2), Posting(1, 1)],
        "dog": [Posting(1, 3)],
    }
    doc_meta = {
        0: DocMeta(0, Path("a.txt"), 10),
        1: DocMeta(1, Path("b.txt"), 30),
    }
    return Index(postings, doc_meta)


def test_len_is_number_of_terms(index):
    assert len(index) == 2


def test_contains(index):
    assert "cat" in index
    assert "python" not in index


def test_getitem_returns_postings(index):
    assert index["dog"] == [Posting(1, 3)]


def test_getitem_missing_raises_keyerror_and_does_not_mutate(index):
    with pytest.raises(KeyError):
        index["python"]
    assert "python" not in index      
    assert len(index) == 2


def test_iter_yields_terms(index):
    assert sorted(index) == ["cat", "dog"]


def test_repr(index):
    assert repr(index) == "Index(terms=2, docs=2)"


def test_num_docs_and_df_and_doc_length(index):
    assert index.num_docs == 2
    assert index.df("cat") == 2
    assert index.df("python") == 0
    assert index.doc_length(1) == 30


def test_avg_doc_length_is_cached(index):
    assert "avg_doc_length" not in index.__dict__
    assert index.avg_doc_length == 20.0
    assert "avg_doc_length" in index.__dict__      


def test_mapping_methods_come_for_free(index):
    assert index.get("python") is None
    assert sorted(index.keys()) == ["cat", "dog"]


def test_open_index_closes_on_normal_exit(index, tmp_path):
    path = tmp_path / "ix.bin"
    save_index(index, path)
    with open_index(path) as ix:
        assert len(ix) == 2
    assert len(ix) == 0                  # close() відпрацював


def test_open_index_closes_even_on_exception(index, tmp_path):
    path = tmp_path / "ix.bin"
    save_index(index, path)
    with pytest.raises(RuntimeError):
        with open_index(path) as ix:
            raise RuntimeError("boom")
    assert len(ix) == 0                  # навіть після винятку


def test_json_roundtrip(index, tmp_path):
    path = tmp_path / "ix.json"
    save_index(index, path)
    with open_index(path) as ix:
        assert ix["cat"] == [Posting(0, 2), Posting(1, 1)]
        assert ix.num_docs == 2