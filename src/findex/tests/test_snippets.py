from findex.evaluate import precision_at_k
from findex.snippets import make_snippet


def test_highlights_match():
    assert make_snippet("The cat sat", ["cat"]) == "The **cat** sat"


def test_empty_when_no_match():
    assert make_snippet("The cat sat", ["dog"]) == ""


def test_window_and_ellipses():
    text = "word " * 100 + "needle " + "word " * 100
    s = make_snippet(text, ["needle"], width=20)
    assert s.startswith("… ") and s.endswith(" …")
    assert "**needle**" in s and len(s) < 80


def test_no_half_words():
    text = "abcdefghij " * 30 + "needle " + "klmnopqrst " * 30
    s = make_snippet(text, ["needle"], width=25)
    for w in s.replace("…", "").replace("**", "").split():
        assert w in {"abcdefghij", "needle", "klmnopqrst"}


def test_prefers_window_with_more_distinct_terms():
    text = "cat " + "x " * 200 + "cat dog " + "y " * 200
    assert "**cat** **dog**" in make_snippet(text, ["cat", "dog"], width=30)


def test_works_on_normalized_text():
    assert make_snippet("Don\u2019t stop", ["don't"]) == "**Don't** stop"


def test_precision_at_k():
    titles = ["moby-dick", "emma", "moby-dick-2", "x", "y"]
    assert precision_at_k(titles, ["moby"]) == 2 / 5
    assert precision_at_k(titles[:2], ["moby"]) == 1 / 5   # ділимо завжди на k