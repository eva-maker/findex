import re
from pathlib import Path
from typing import Iterator


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


if __name__ == "__main__":
    for doc in iter_documents(Path("data/raw")):
        print(len(doc), repr(doc[:80]))