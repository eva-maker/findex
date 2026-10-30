"""Булеві операції над відсортованими списками doc_id (Лаба 2): merge і set.

Це бібліотечний модуль. Командний рядок тепер у findex.cli (команда `findex search`).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from findex.models import Posting

Engine = Literal["merge", "set"]
Op = Literal["AND", "OR", "NOT"]
type IdList = list[int]


def doc_ids(postings: Sequence[Posting]) -> IdList:
    """Postings уже відсортовані за doc_id (це гарантія build_index) —
    тут просто дістаємо самі id."""
    return [p.doc_id for p in postings]


# двовказівниковий merge по відсортованих списках


def merge_and(a: IdList, b: IdList) -> IdList:
    result: IdList = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def merge_or(a: IdList, b: IdList) -> IdList:
    result: IdList = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


def merge_not(a: IdList, b: IdList) -> IdList:
    """a NOT b: doc_id з a, яких немає в b."""
    result: IdList = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            result.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return result


# альтернатива через set


def set_and(a: IdList, b: IdList) -> IdList:
    return sorted(set(a) & set(b))


def set_or(a: IdList, b: IdList) -> IdList:
    return sorted(set(a) | set(b))


def set_not(a: IdList, b: IdList) -> IdList:
    return sorted(set(a) - set(b))


_ENGINES: dict[Engine, dict[Op, Callable[[IdList, IdList], IdList]]] = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def evaluate_query(
    query: str,
    postings: Mapping[str, Sequence[Posting]],
    engine: Engine = "merge",
) -> IdList:
    """Проста ліва-направо оцінка: 'a b' = AND, явні OR / NOT між термінами."""
    ops = _ENGINES[engine]
    tokens = query.split()
    if not tokens:
        return []

    def get(term: str) -> IdList:
        return doc_ids(postings.get(term.lower(), []))

    result = get(tokens[0])
    i = 1
    while i < len(tokens):
        word = tokens[i].upper()
        if word == "OR" or word == "NOT":
            i += 1
            result = ops[word](result, get(tokens[i]))
        else:
            result = ops["AND"](result, get(tokens[i]))
        i += 1
    return result
