"""Токенізатор: таблиця прикладів (parametrize), включно з Unicode."""

import pytest

from findex.tokens import normalize_text, token_spans, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("   \n\t ", []),
        ("!!! ??? ...", []),
        ("Hello, World!", ["hello", "world"]),
        ("ПРИВІТ світе", ["привіт", "світе"]),  # кирилиця + нижній регістр
        ("Ünï-côdé", ["ünï-côdé"]),  # діакритика й дефіс
        ("ﬁne", ["fine"]),  # лігатура ﬁ -> fi (NFKC)
        ("ｆｕｌｌ ｗｉｄｔｈ", ["full", "width"]),  # повноширинні літери -> звичайні
        ("don‘t", ["don't"]),  # різні «криві» апострофи
        ("don‛t", ["don't"]),
        ("don′t", ["don't"]),
        ("3.14 and 1,000,000", ["3.14", "and", "1,000,000"]),
        ("snake_case", ["snake", "case"]),  # підкреслення — розмітка, не літера
    ],
)
def test_tokenize_table(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_is_lazy_generator() -> None:
    tokens = tokenize("one two three")
    assert next(tokens) == "one"  # Iterator: можна брати по одному


def test_token_spans_point_into_normalized_text() -> None:
    text = normalize_text("Hello, brave new world")
    for token, start, end in token_spans(text):
        assert text[start:end].lower() == token


def test_normalize_text_is_idempotent() -> None:
    once = normalize_text("Don’t ﬁnd")
    assert normalize_text(once) == once


def test_dotted_capital_i_survives_retokenization() -> None:
    """Регресія: «İ» у нижньому регістрі = «i» + комбінівна крапка. Раніше другий прохід
    токенізації відкидав крапку, і tokenize(join(tokenize(x))) != tokenize(x)."""
    once = list(tokenize("İ"))
    assert once == ["i̇"]
    assert list(tokenize(" ".join(once))) == once
