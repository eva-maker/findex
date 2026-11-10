import re
import unicodedata
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

_APOSTROPHE_VARIANTS = ("\u2018", "\u2019", "\u201b", "\u2032")


# Літера = будь-яка літера Unicode АБО комбінівний діакритичний знак (U+0300–U+036F).
# Другий варіант потрібен, бо .lower() для «İ» (турецька I з крапкою) дає
# «i» + комбінівну крапку U+0307: без нього токен розвалювався б при повторній
# токенізації. Це контрприклад до властивості ідемпотентності (tests/test_properties.py).
_LETTER = r"(?:[^\W\d_]|[\u0300-\u036f])"

_TOKEN_RE = re.compile(
    rf"""
    \d+(?:[.,]\d+)*
    | {_LETTER}+(?:['\-]{_LETTER}+)*
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


def token_table(text: str) -> tuple[list[str], NDArray[np.int32]]:
    matches = list(_TOKEN_RE.finditer(text))
    tokens = [m.group().lower() for m in matches]
    spans = np.array([m.span() for m in matches], dtype=np.int32).reshape(-1, 2)
    return tokens, spans


def tokenize(text: str) -> Iterator[str]:
    yield from map(str.lower, _TOKEN_RE.findall(_normalize(text)))
