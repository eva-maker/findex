"""Merge-алгоритми по відсортованих списках і їхня «еталонна» версія через set."""

from collections.abc import Callable

import pytest

from findex.models import Index
from findex.search import (
    Engine,
    evaluate_query,
    merge_and,
    merge_not,
    merge_or,
    set_and,
    set_not,
    set_or,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 3, 7], [3, 7, 9], [3, 7]),
        ([], [1, 2], []),
        ([1, 2], [], []),
        ([1, 2, 3], [4, 5], []),
        ([2, 4], [1, 2, 3, 4, 5], [2, 4]),
    ],
)
def test_merge_and(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_and(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 3, 7], [3, 7, 9], [1, 3, 7, 9]),
        ([], [1, 2], [1, 2]),
        ([1, 2], [], [1, 2]),
        ([1, 5], [2, 3], [1, 2, 3, 5]),
    ],
)
def test_merge_or(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_or(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 3, 7], [3, 7, 9], [1]),
        ([1, 2], [], [1, 2]),
        ([], [1], []),
        ([1, 2, 3], [1, 2, 3], []),
    ],
)
def test_merge_not(a: list[int], b: list[int], expected: list[int]) -> None:
    assert merge_not(a, b) == expected


@pytest.mark.parametrize(
    ("merge_fn", "set_fn"),
    [(merge_and, set_and), (merge_or, set_or), (merge_not, set_not)],
)
def test_merge_agrees_with_set_on_an_example(
    merge_fn: Callable[[list[int], list[int]], list[int]],
    set_fn: Callable[[list[int], list[int]], list[int]],
) -> None:
    a, b = [1, 4, 6, 9], [2, 4, 9, 10]
    assert merge_fn(a, b) == set_fn(a, b)


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cat", [0, 1]),
        ("cat dog", [1]),  # неявний AND
        ("sat OR dog", [0, 1]),
        ("cat NOT dog", [0]),
        ("zebra", []),
        ("", []),
    ],
)
def test_evaluate_query_engines_agree(
    tiny_index: Index, engine: Engine, query: str, expected: list[int]
) -> None:
    assert evaluate_query(query, tiny_index, engine=engine) == expected
