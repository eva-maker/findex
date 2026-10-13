import argparse
import json
from pathlib import Path

from findex.ranking import BM25, TfIdf, search
from findex.serialize import open_index


def precision_at_k(titles: list[str], relevant_fragments: list[str], k: int = 5) -> float:
    """Скільки з топ-k вважаємо правильними / k (ділимо завжди на k)."""
    frags = [f.lower() for f in relevant_fragments]
    good = sum(any(f in t.lower() for f in frags) for t in titles[:k])
    return good / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_path", type=Path)
    ap.add_argument("queries", type=Path)
    args = ap.parse_args()
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    scorers = {"TF-IDF": TfIdf(), "BM25": BM25()}
    totals = dict.fromkeys(scorers, 0.0)
    print("| Запит | " + " | ".join(scorers) + " |")
    print("|---|" + "---:|" * len(scorers))
    with open_index(args.index_path) as index:
        for case in cases:
            row = []
            for name, scorer in scorers.items():
                titles = [r.title for r in search(index, case["query"], scorer, k=5)]
                p = precision_at_k(titles, case["relevant"])
                totals[name] += p
                row.append(f"{p:.2f}")
            print(f"| {case['query']} | " + " | ".join(row) + " |")
    print("| **середнє** | " + " | ".join(f"{totals[n] / len(cases):.2f}" for n in scorers) + " |")


if __name__ == "__main__":
    main()