import random
from pathlib import Path

import pytest

from findex.corpus import read_document
from findex.snippets import (
    Match,
    _best_center,  # pyright: ignore[reportPrivateUsage]
    load_doc_tokens,
    make_snippet,
    snippet_from_tokens,
    warm_doc_tokens,
)


def best_center_quadratic(matches: list[Match], width: int) -> int:
    best_start, best_score = matches[0][1], -1
    for _t, start, _e in matches:
        score = len({t for t, s, _ in matches if start <= s <= start + width})
        if score > best_score:
            best_start, best_score = start, score
    return best_start


@pytest.mark.parametrize("seed", range(30))
def test_best_center_same_as_quadratic_version(seed: int) -> None:
    rng = random.Random(seed)
    position = 0
    matches: list[Match] = []
    for _ in range(rng.randint(1, 60)):
        position += rng.randint(2, 40)
        matches.append((rng.choice("abcd"), position, position + 1))
    width = rng.choice([1, 10, 40, 80, 300])
    assert _best_center(matches, width) == best_center_quadratic(matches, width)


@pytest.mark.parametrize("width", [20, 80])
@pytest.mark.parametrize("terms", [["aaa"], ["aab", "aac"], ["aaa", "aad", "aaz"], ["nothere"]])
def test_fast_snippet_equals_slow_on_synthetic(
    synthetic_corpus: Path, terms: list[str], width: int
) -> None:
    for path in sorted(synthetic_corpus.glob("*.txt"))[:6]:
        slow = make_snippet(read_document(path), terms, width=width, mark="\x00")
        fast = snippet_from_tokens(load_doc_tokens(path), terms, width=width, mark="\x00")
        assert fast == slow


def test_fast_snippet_equals_slow_on_tiny(tiny_corpus: Path) -> None:
    (tiny_corpus / "d.txt").write_text(
        "Don’t stop. Don't go, state-of-the-art 3.14 İstanbul.", encoding="utf-8"
)
    for path in sorted(tiny_corpus.glob("*.txt")):
        for terms in (["cat"], ["old", "man"], ["don't"], ["state-of-the-art"], ["3.14"], ["x"]):
            assert snippet_from_tokens(load_doc_tokens(path), terms) == make_snippet(
                read_document(path), terms
            )


def test_warm_doc_tokens_skips_missing(tiny_corpus: Path, tmp_path: Path) -> None:
    paths = [*sorted(tiny_corpus.glob("*.txt")), tmp_path / "missing.txt"]
    assert warm_doc_tokens(paths) == 3
