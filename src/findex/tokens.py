import re
import unicodedata
from collections.abc import Iterator

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


def tokenize(text: str) -> Iterator[str]:
    for token, _start, _end in token_spans(_normalize(text)):
        yield token
