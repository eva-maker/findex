import random
from pathlib import Path

import pytest

from findex.index import build_index
from findex.models import Index


def pseudo_word(i: int) -> str:
    return chr(97 + i // 676 % 26) + chr(97 + i // 26 % 26) + chr(97 + i % 26)


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = random.Random(7)
    vocab = [pseudo_word(i) for i in range(800)]
    weights = [1 / (rank + 1) for rank in range(len(vocab))]
    corpus = tmp_path_factory.mktemp("synthetic")
    for number in range(24):
        words = rng.choices(vocab, weights, k=4000)
        (corpus / f"doc{number:02d}.txt").write_text(" ".join(words) + ".", encoding="utf-8")
    return corpus


@pytest.fixture(scope="session")
def synthetic_index(synthetic_corpus: Path) -> Index:
    return build_index(synthetic_corpus)
