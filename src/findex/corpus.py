import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_START_RE = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def _strip_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg license header/footer, keep only the book text."""
    m = _START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _END_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def read_document(path: Path) -> str:
    """Прочитати один файл і зрізати службовий текст Gutenberg.
    Потрібна окремо, щоб пошук міг дістати текст для сніпета."""
    return _strip_boilerplate(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class Document:
    path: Path
    text: str


def iter_documents_with_meta(corpus_dir: Path) -> Iterator[Document]:
    for path in sorted(corpus_dir.glob("*.txt")):
        yield Document(path=path, text=read_document(path))


def iter_documents(corpus_dir: Path) -> Iterator[str]:
    for doc in iter_documents_with_meta(corpus_dir):
        yield doc.text