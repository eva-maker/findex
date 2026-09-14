import re
import unicodedata
from typing import Iterator


_APOSTROPHE_VARIANTS = ("\u2018", "\u2019", "\u201b", "\u2032")


_TOKEN_RE = re.compile(
    r"""
    \d+(?:[.,]\d+)*
    | [^\W\d_]+(?:['\-][^\W\d_]+)*
    """,
    re.VERBOSE,
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in _APOSTROPHE_VARIANTS:
        text = text.replace(ch, "'")
    return text


def tokenize(text: str) -> Iterator[str]:

    normalized = _normalize(text)
    for match in _TOKEN_RE.finditer(normalized):
        yield match.group().lower()