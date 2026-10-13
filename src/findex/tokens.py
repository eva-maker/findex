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


def normalize_text(text: str) -> str:
    """Той самий нормалізований текст, на якому працює tokenize()."""
    return _normalize(text)


def token_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """(токен, початок, кінець) для УЖЕ нормалізованого тексту."""
    for match in _TOKEN_RE.finditer(text):
        yield match.group().lower(), match.start(), match.end()


def tokenize(text: str) -> Iterator[str]:
    for token, _start, _end in token_spans(_normalize(text)):
        yield token