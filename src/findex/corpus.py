import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_START_RE = re.compile(
    r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE
)
_END_RE = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def _strip_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg license header/footer, keep only the book text."""
    m = _START_RE.search(text)
    if m:
        text = text[m.end() :]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def read_document(path: Path) -> str:
    """Прочитати один файл і зрізати службовий текст Gutenberg.
    Потрібна окремо, щоб пошук міг дістати текст для сніпета."""
    return _strip_boilerplate(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class Document:
    path: Path
    text: str


def list_document_paths(corpus_dir: Path) -> list[Path]:
    """Усі .txt у каталозі в стабільному (відсортованому) порядку: від нього залежить doc_id."""
    return sorted(corpus_dir.glob("*.txt"))


def load_document(path: Path) -> Document:
    return Document(path=path, text=read_document(path))


def iter_documents_with_meta(corpus_dir: Path) -> Iterator[Document]:
    for path in list_document_paths(corpus_dir):
        yield load_document(path)


def iter_jsonl_documents(jsonl_path: Path) -> Iterator[Document]:
    """Сторінки краулера (Лаба 6): один JSON-рядок = одна сторінка. Читаємо потоком, по рядку."""
    with Path(jsonl_path).open(encoding="utf-8") as file:
        for number, line in enumerate(file):
            if line.strip():
                record = json.loads(line)
                title = str(record.get("title", ""))
                text = str(record.get("text", ""))
                yield Document(
                    path=Path(page_filename(number, str(record["url"]))),
                    text=f"{title}\n\n{text}".strip(),
                )


def page_filename(number: int, url: str) -> str:
    """Ім'я файла для сторінки: порядковий номер + скорочена адреса, напр. ``0007-p-7.txt``."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{parts.path}-{parts.query}").strip("-").lower()[:60]
    return f"{number:04d}-{slug or 'index'}.txt"


def unpack_jsonl(jsonl_path: Path, dest_dir: Path) -> int:
    """Розкласти JSONL по .txt-файлах, щоб решта findex (індекс, сніпети) працювала як раніше.

    Стирає лише власні файли ``NNNN-*.txt`` з минулого розпакування й пише ``urls.tsv``
    (файл -> адреса). Повертає кількість сторінок.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("[0-9][0-9][0-9][0-9]-*.txt"):
        old.unlink()
    count = 0
    mapping: list[str] = []
    with Path(jsonl_path).open(encoding="utf-8") as file:
        for number, line in enumerate(file):
            if not line.strip():
                continue
            record = json.loads(line)
            name = page_filename(number, str(record["url"]))
            title, text = str(record.get("title", "")), str(record.get("text", ""))
            (dest_dir / name).write_text(f"{title}\n\n{text}".strip() + "\n", encoding="utf-8")
            mapping.append(f"{name}\t{record['url']}")
            count += 1
    (dest_dir / "urls.tsv").write_text("\n".join(mapping) + "\n", encoding="utf-8")
    return count


def iter_documents(corpus: Path) -> Iterator[str]:
    """Тексти документів: з каталогу .txt (Лаба 1) або з JSONL краулера (Лаба 6)."""
    corpus = Path(corpus)
    if corpus.is_file():
        for doc in iter_jsonl_documents(corpus):
            yield doc.text
        return
    for doc in iter_documents_with_meta(corpus):
        yield doc.text
