"""Precision@5 для режимів пошуку: BM25 (AND), BM25 (OR), semantic, hybrid.

    python -m findex.evaluate_modes index.bin embeddings.npz eval/queries.json eval/queries_paraphrase.json

Для кожного файлу запитів друкується markdown-таблиця (вставляй у README):
у дужках — місце першого правильного документа в топі (з одним правильним P@5 ≤ 0.20).
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from findex.evaluate import Case, precision_at_k
from findex.models import Index
from findex.query import QueryError
from findex.semantic import (
    Embedder,
    FastEmbedder,
    SemanticIndex,
    keyword_query,
    rrf,
    semantic_rank,
)
from findex.serialize import open_index
from findex.vectorized import search_vectorized

TOP = 5


def _titles(index: Index, doc_ids: list[int]) -> list[str]:
    return [index.doc(d).path.stem for d in doc_ids[:TOP]]


def _keyword(index: Index, query: str) -> list[int]:
    try:
        return [r.doc_id for r in search_vectorized(index, query, "bm25", index.num_docs)]
    except QueryError:
        return []


def first_relevant(titles: list[str], fragments: list[str]) -> str:
    """Місце першого правильного документа в топі (чесніше за P@5, коли правильний лише один)."""
    frags = [f.lower() for f in fragments]
    for place, title in enumerate(titles, start=1):
        if any(f in title.lower() for f in frags):
            return f"#{place}"
    return "–"


def rankers(
    index: Index, sem: SemanticIndex, embedder: Embedder
) -> dict[str, Callable[[str], list[int]]]:
    def semantic(q: str) -> list[int]:
        return [h.doc_id for h in semantic_rank(sem, embedder, q)]

    def or_query(q: str) -> list[int]:
        return _keyword(index, keyword_query(q)) if keyword_query(q) else []

    def hybrid(q: str) -> list[int]:
        return [d for d, _ in rrf([semantic(q), or_query(q)])]

    return {
        "BM25 (AND)": lambda q: _keyword(index, q),
        "BM25 (OR)": or_query,
        "semantic": semantic,
        "hybrid": hybrid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_path", type=Path)
    ap.add_argument("embeddings", type=Path)
    ap.add_argument("queries", type=Path, nargs="+")
    args = ap.parse_args()
    sem = SemanticIndex.load(args.embeddings)
    embedder = FastEmbedder(sem.model)
    with open_index(args.index_path) as index:
        ranked = rankers(index, sem, embedder)
        for path in args.queries:
            cases: list[Case] = json.loads(path.read_text(encoding="utf-8"))
            totals = dict.fromkeys(ranked, 0.0)
            print(f"\n### {path.name}\n")
            print("| Запит | " + " | ".join(ranked) + " |")
            print("|---|" + "---:|" * len(ranked))
            for case in cases:
                row: list[str] = []
                for name, rank in ranked.items():
                    titles = _titles(index, rank(case["query"]))
                    p = precision_at_k(titles, case["relevant"], TOP)
                    totals[name] += p
                    row.append(f"{p:.2f} ({first_relevant(titles, case['relevant'])})")
                print(f"| {case['query']} | " + " | ".join(row) + " |")
            means = " | ".join(f"{totals[n] / len(cases):.2f}" for n in ranked)
            print(f"| **середнє** | {means} |")


if __name__ == "__main__":
    main()
