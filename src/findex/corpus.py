import re
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass


_START_RE = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def _strip_boilerplate(text: str) -> str:
    start_match = _START_RE.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = _END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()


def iter_documents(corpus_dir: Path) -> Iterator[str]:
    for path in sorted(corpus_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        yield _strip_boilerplate(raw_text)


@dataclass(frozen=True, slots=True)
class Document:
    path: Path
    text: str


def iter_documents_with_meta(corpus_dir: Path) -> Iterator[Document]:
    """Yield Document(path, text) for each .txt file, one at a time."""
    for path in sorted(corpus_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        yield Document(path=path, text=_strip_boilerplate(raw_text))

#
def iter_documents(corpus_dir: Path) -> Iterator[str]:
    """Yield just the cleaned text (kept for lab 1 compatibility)."""
    for doc in iter_documents_with_meta(corpus_dir):
        yield doc.text


if __name__ == "__main__":
    for doc in iter_documents(Path("data/raw")):
        print(len(doc), repr(doc[:80]))