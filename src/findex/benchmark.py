import time
from pathlib import Path

from findex.index import build_index
from findex.search import evaluate_query


def find_rarest_terms(postings: dict, n: int = 2) -> list[str]:
    real_words = [term for term in postings if term.isalpha() and len(term) > 2]
    return sorted(real_words, key=lambda term: (len(postings[term]), term))[:n]


def timed_query(query: str, postings: dict, engine: str, repeats: int = 5) -> float:
    """Середній час одного виконання запиту за кілька повторів (щоб згладити шум)."""
    start = time.perf_counter()
    for _ in range(repeats):
        evaluate_query(query, postings, engine=engine)
    elapsed = time.perf_counter() - start
    return elapsed / repeats


if __name__ == "__main__":
    postings, doc_meta = build_index(Path("data/raw"))

    frequent_terms = ["the", "of"]
    rare_terms = find_rarest_terms(postings, n=2)
    print(f"Найрідкісніші терміни: {rare_terms}")

    queries = {
        "frequent AND": f"{frequent_terms[0]} {frequent_terms[1]}",
        "frequent OR": f"{frequent_terms[0]} OR {frequent_terms[1]}",
        "rare AND": f"{rare_terms[0]} {rare_terms[1]}",
        "rare OR": f"{rare_terms[0]} OR {rare_terms[1]}",
    }

    print(f"\n{'Запит':<15} {'merge (мс)':>12} {'set (мс)':>12}")
    for label, query in queries.items():
        merge_time = timed_query(query, postings, "merge") * 1000
        set_time = timed_query(query, postings, "set") * 1000
        print(f"{label:<15} {merge_time:>12.4f} {set_time:>12.4f}")