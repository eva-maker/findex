from pathlib import Path

import pytest

from findex.models import DocMeta, Index, Posting
from findex.query import And, Not, Or, Phrase, QueryError, Term, parse
from findex.ranking import search


# дерева (парсер)

def test_single_term():
    assert parse("a") == Term("a")


def test_implicit_and():
    assert parse("a b") == And(Term("a"), Term("b"))


def test_explicit_and():
    assert parse("a AND b") == And(Term("a"), Term("b"))


def test_and_binds_tighter_than_or():
    # 'a OR b c' — це a OR (b AND c), а не (a OR b) AND c
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_parentheses_override_precedence():
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))


def test_not_applies_to_next_atom():
    assert parse("a NOT b") == And(Term("a"), Not(Term("b")))


def test_phrase():
    assert parse('"event loop"') == Phrase(("event", "loop"))


def test_one_word_phrase_is_just_a_term():
    assert parse('"python"') == Term("python")


def test_full_example_from_the_lab():
    tree = parse('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert tree == expected


def test_words_are_normalized_like_documents():
    assert parse("Python") == Term("python")


def test_lowercase_operators_are_ordinary_words():
    assert parse("a or b") == And(And(Term("a"), Term("or")), Term("b"))


def test_operator_overloading_builds_the_same_tree():
    built = Term("a") & (Term("b") | Term("c")) & ~Term("d")
    assert built == parse("a (b OR c) NOT d")


@pytest.mark.parametrize("bad", ["", "   ", "(a", "a)", "a AND", "a OR OR b", '"event loop', '""', "NOT"])
def test_bad_queries_raise_query_error(bad):
    with pytest.raises(QueryError):
        parse(bad)


# виконання

@pytest.fixture
def index() -> Index:
    # doc0: event loop python | doc1: loop event python
    # doc2: python async      | doc3: java event loop
    postings = {
        "event": [Posting(0, 1, (0,)), Posting(1, 1, (1,)), Posting(3, 1, (1,))],
        "loop": [Posting(0, 1, (1,)), Posting(1, 1, (0,)), Posting(3, 1, (2,))],
        "python": [Posting(0, 1, (2,)), Posting(1, 1, (2,)), Posting(2, 1, (0,))],
        "async": [Posting(2, 1, (1,))],
        "java": [Posting(3, 1, (0,))],
    }
    lengths = {0: 3, 1: 3, 2: 2, 3: 3}
    doc_meta = {d: DocMeta(d, Path(f"doc{d}.txt"), n) for d, n in lengths.items()}
    return Index(postings, doc_meta)


def test_term_and_and_or(index):
    assert parse("event").evaluate(index) == {0, 1, 3}
    assert parse("event loop").evaluate(index) == {0, 1, 3}
    assert parse("async OR java").evaluate(index) == {2, 3}


def test_not(index):
    assert parse("python NOT async").evaluate(index) == {0, 1}
    assert Not(Term("event")).evaluate(index) == {2}


def test_parentheses_in_evaluation(index):
    assert parse("(async OR java) event").evaluate(index) == {3}


def test_phrase_respects_word_order(index):
    # у doc1 слова є, але в іншому порядку: loop event
    assert parse('"event loop"').evaluate(index) == {0, 3}
    assert parse('"loop event"').evaluate(index) == {1}


def test_phrase_with_unknown_word_matches_nothing(index):
    assert parse('"event zebra"').evaluate(index) == set()


def test_phrase_needs_positions():
    no_positions = Index(
        {"a": [Posting(0, 1)], "b": [Posting(0, 1)]},
        {0: DocMeta(0, Path("x.txt"), 2)},
    )
    with pytest.raises(QueryError):
        parse('"a b"').evaluate(no_positions)


def test_has_positions_flag(index):
    assert index.has_positions is True
    plain = Index({"a": [Posting(0, 1)]}, {0: DocMeta(0, Path("x.txt"), 1)})
    assert plain.has_positions is False


def test_ranking_uses_the_query_language(index):
    results = search(index, '"event loop"')
    assert {r.doc_id for r in results} == {0, 3}


def test_negated_terms_do_not_score(index):
    results = search(index, "python NOT async")
    assert {r.doc_id for r in results} == {0, 1}