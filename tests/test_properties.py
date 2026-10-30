"""Властивості (Hypothesis): не окремі приклади, а твердження «для будь-якого входу».

Коли Hypothesis знаходить контрприклад, він сам стискає його до найменшого
(shrinking) — тому повідомлення про помилку коротке й зрозуміле.
"""

import tempfile
from pathlib import Path

from hypothesis import example, given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import index_documents
from findex.models import DocMeta, Index, Posting
from findex.search import merge_and, merge_not, merge_or, set_and, set_not, set_or
from findex.serialize import load_index, save_index
from findex.tokens import tokenize

# Без сурогатів (Cs): це «половинки» символів, які не можна записати в UTF-8.
text_strategy = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200)

# відсортований список унікальних doc_id — такими є реальні постінги
sorted_ids = st.lists(st.integers(min_value=0, max_value=200), unique=True).map(sorted)


@st.composite
def indexes(draw: st.DrawFn) -> Index:
    """Випадковий, але коректний Index: відсортовані постінги, довжини > 0."""
    num_docs = draw(st.integers(min_value=1, max_value=5))
    with_positions = draw(st.booleans())
    terms = draw(
        st.lists(
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=8),
            max_size=8,
            unique=True,
        )
    )
    postings: dict[str, list[Posting]] = {}
    for term in terms:
        doc_ids = draw(
            st.lists(st.integers(0, num_docs - 1), min_size=1, max_size=num_docs, unique=True)
        )
        plist: list[Posting] = []
        for doc_id in sorted(doc_ids):
            freq = draw(st.integers(min_value=1, max_value=5))
            positions = tuple(range(freq)) if with_positions else ()
            plist.append(Posting(doc_id, freq, positions))
        postings[term] = plist
    doc_meta = {
        d: DocMeta(d, Path(f"doc{d}.txt"), draw(st.integers(min_value=1, max_value=1000)))
        for d in range(num_docs)
    }
    return Index(postings, doc_meta)


def snapshot(index: Index) -> tuple[dict[str, list[Posting]], list[DocMeta]]:
    return {t: list(pl) for t, pl in index.items()}, list(index.docs())


@settings(max_examples=50, deadline=None)
@given(index=indexes(), fmt=st.sampled_from([".bin", ".json"]))
def test_property_save_load_roundtrip(index: Index, fmt: str) -> None:
    """load(save(index)) містить рівно те саме, що й index (обидва формати)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"ix{fmt}"
        save_index(index, path)
        assert snapshot(load_index(path)) == snapshot(index)


@settings(max_examples=100)
@given(texts=st.lists(text_strategy, max_size=6))
def test_property_postings_are_always_sorted(texts: list[str]) -> None:
    """Для будь-яких документів списки постінгів впорядковані за doc_id без дублікатів."""
    docs = [Document(Path(f"{i}.txt"), text) for i, text in enumerate(texts)]
    index = index_documents(docs)
    for plist in index.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids))
        assert all(p.freq >= 1 for p in plist)


@given(a=sorted_ids, b=sorted_ids)
def test_property_merge_matches_set_operations(a: list[int], b: list[int]) -> None:
    """«Оракул»: швидкий merge збігається з очевидно правильним set."""
    assert merge_and(a, b) == sorted(set(a) & set(b)) == set_and(a, b)
    assert merge_or(a, b) == sorted(set(a) | set(b)) == set_or(a, b)
    assert merge_not(a, b) == sorted(set(a) - set(b)) == set_not(a, b)


@given(text=text_strategy)
def test_property_tokens_are_never_empty(text: str) -> None:
    assert all(token != "" for token in tokenize(text))


@example("İ")  # справжній контрприклад, який Hypothesis сам би знайшов не одразу
@given(text=text_strategy)
def test_property_tokenize_is_idempotent(text: str) -> None:
    """Токенізація вже токенізованого тексту нічого не змінює."""
    once = list(tokenize(text))
    assert list(tokenize(" ".join(once))) == once
