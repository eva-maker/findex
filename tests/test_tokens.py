from findex.tokens import tokenize


def test_simple_words() -> None:
    assert list(tokenize("Hello world")) == ["hello", "world"]


def test_straight_apostrophe_contraction() -> None:
    assert list(tokenize("don't stop")) == ["don't", "stop"]


def test_curly_apostrophe_normalizes_to_straight() -> None:
    assert list(tokenize("don\u2019t stop")) == ["don't", "stop"]


def test_leading_apostrophe_is_not_part_of_token() -> None:
    # quotation mark, not a contraction
    assert list(tokenize("'hello' she said")) == ["hello", "she", "said"]


def test_hyphenated_compound_kept_together() -> None:
    assert list(tokenize("state-of-the-art design")) == ["state-of-the-art", "design"]


def test_dash_as_punctuation_does_not_join_words() -> None:
    assert list(tokenize("wait - stop")) == ["wait", "stop"]


def test_numbers_with_separators() -> None:
    assert list(tokenize("1,000 or 3.14")) == ["1,000", "or", "3.14"]


def test_underscored_italics_markup_is_stripped() -> None:
    # Gutenberg plain text uses _word_ for italics
    assert list(tokenize("_Ask_ him")) == ["ask", "him"]


def test_accented_letters() -> None:
    assert list(tokenize("l'institut français")) == ["l'institut", "français"]
