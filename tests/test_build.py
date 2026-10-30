"""Побудова індексу на маленькому корпусі з conftest.py."""

from collections.abc import Iterator
from pathlib import Path

from findex.corpus import Document, iter_documents, list_document_paths, read_document
from findex.index import build_index, index_documents
from findex.models import Index, Posting


def test_terms_and_frequencies(tiny_index: Index) -> None:
    assert tiny_index["cat"] == [Posting(0, 2), Posting(1, 1)]
    assert tiny_index["sea"] == [Posting(2, 1)]
    assert tiny_index.df("cat") == 2
    assert tiny_index.df("zebra") == 0


def test_doc_lengths_count_tokens(tiny_index: Index) -> None:
    assert [tiny_index.doc_length(d) for d in range(3)] == [6, 5, 9]


def test_doc_ids_follow_file_name_order(tiny_index: Index) -> None:
    assert [m.path.name for m in tiny_index.docs()] == ["a.txt", "b.txt", "c.txt"]


def test_postings_are_sorted_by_doc_id(tiny_index: Index) -> None:
    for plist in tiny_index.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)


def test_positions_are_recorded_when_requested(tiny_index_pos: Index) -> None:
    # "the cat sat the cat ran": cat на позиціях 1 і 4
    assert tiny_index_pos["cat"][0].positions == (1, 4)
    assert tiny_index_pos.has_positions is True


def test_no_positions_by_default(tiny_index: Index) -> None:
    assert tiny_index["cat"][0].positions == ()
    assert tiny_index.has_positions is False


def test_index_documents_accepts_a_stream() -> None:
    def stream() -> Iterator[Document]:
        yield Document(Path("x.txt"), "alpha beta")

    index = index_documents(stream())
    assert sorted(index) == ["alpha", "beta"]


def test_empty_corpus_gives_empty_index() -> None:
    index = index_documents([])
    assert len(index) == 0
    assert index.num_docs == 0
    assert index.has_positions is False


def test_build_index_from_directory_matches_in_memory(tiny_corpus: Path, tiny_index: Index) -> None:
    from_disk = build_index(tiny_corpus)
    assert {t: list(p) for t, p in from_disk.items()} == {t: list(p) for t, p in tiny_index.items()}


def test_list_document_paths_is_sorted_and_only_txt(tiny_corpus: Path) -> None:
    (tiny_corpus / "notes.md").write_text("ignored", encoding="utf-8")
    assert [p.name for p in list_document_paths(tiny_corpus)] == ["a.txt", "b.txt", "c.txt"]


def test_gutenberg_boilerplate_is_stripped(tmp_path: Path) -> None:
    book = tmp_path / "book.txt"
    book.write_text(
        "License blah\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nReal text here.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nMore license",
        encoding="utf-8",
    )
    assert read_document(book) == "Real text here."


def test_iter_documents_yields_texts(tiny_corpus: Path) -> None:
    texts = list(iter_documents(tiny_corpus))
    assert len(texts) == 3
    assert texts[1] == "A dog and a cat."
