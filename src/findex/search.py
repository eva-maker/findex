from findex.models import Posting

import argparse
import time
import tracemalloc
from pathlib import Path

from findex.serialize import load_json, load_pickle


def doc_ids(postings: list[Posting]) -> list[int]:
    """Postings уже відсортовані за doc_id (це гарантія build_index) —
    тут просто дістаємо самі id."""
    return [p.doc_id for p in postings]


# двовказівниковий merge по відсортованих списках 

def merge_and(a: list[int], b: list[int]) -> list[int]:
    result = []
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


def merge_or(a: list[int], b: list[int]) -> list[int]:
    result = []
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


def merge_not(a: list[int], b: list[int]) -> list[int]:
    """a NOT b: doc_id з a, яких немає в b."""
    result = []
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

def set_and(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_not(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) - set(b))


_ENGINES = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def evaluate_query(query: str, postings: dict, engine: str = "merge") -> list[int]:
    """Проста ліва-направо оцінка: 'a b' = AND, явні OR / NOT між термінами."""
    ops = _ENGINES[engine]
    tokens = query.split()
    if not tokens:
        return []

    def get(term: str) -> list[int]:
        return doc_ids(postings.get(term.lower(), []))

    result = get(tokens[0])
    i = 1
    while i < len(tokens):
        tok = tokens[i].upper()
        if tok in ("OR", "NOT"):
            i += 1
            term_ids = get(tokens[i])
            result = ops[tok](result, term_ids)
        else:
            term_ids = get(tokens[i])
            result = ops["AND"](result, term_ids)
        i += 1
    return result


def _load_index(path: Path):
    if path.suffix == ".json":
        return load_json(path)
    return load_pickle(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a saved inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to a saved index file")
    parser.add_argument("query", type=str, help='e.g. "cat dog", "cat OR dog", "cat NOT dog"')
    parser.add_argument("--engine", choices=("merge", "set"), default="merge")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()

    postings, doc_meta = _load_index(args.index_path)
    result_ids = evaluate_query(args.query, postings, engine=args.engine)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query: {args.query!r} (engine={args.engine})")
    print(f"Found {len(result_ids)} document(s):")
    for doc_id in result_ids:
        meta = doc_meta[doc_id]
        print(f"  [{doc_id}] {meta.path.name} ({meta.length} tokens)")

    print(f"\nTime:        {elapsed:.4f}s")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()