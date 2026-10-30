"""Спільні фікстури: маленький корпус із трьох книжок і індекс, збудований з нього.

Усе в пам'яті або в tmp_path: без мережі й без реального корпусу з data/.
"""

from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index, index_documents
from findex.models import Index

# Три «книжки» з передбачуваним вмістом. Номери doc_id = порядок за іменем файлу.
TINY_TEXTS: dict[str, str] = {
    "a.txt": "The cat sat. The cat ran.",  # cat x2, sat, ran, the x2
    "b.txt": "A dog and a cat.",  # dog, cat
    "c.txt": "The old man and the sea. The old man.",  # old man x2, sea
}


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """Каталог із трьома .txt-файлами."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in TINY_TEXTS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


@pytest.fixture
def tiny_docs() -> list[Document]:
    return [Document(Path(name), text) for name, text in TINY_TEXTS.items()]


@pytest.fixture
def tiny_index(tiny_docs: list[Document]) -> Index:
    """Індекс без позицій, збудований у пам'яті."""
    return index_documents(tiny_docs)


@pytest.fixture
def tiny_index_pos(tiny_docs: list[Document]) -> Index:
    return index_documents(tiny_docs, positions=True)


@pytest.fixture
def tiny_index_file(tiny_corpus: Path, tmp_path: Path) -> Path:
    """Збережений на диск індекс з позиціями (для CLI-тестів)."""
    from findex.serialize import save_index

    path = tmp_path / "tiny.bin"
    save_index(build_index(tiny_corpus, positions=True), path)
    return path
