"""Три sanity-check рейтингу на справжньому корпусі (для README).

Запуск: uv run python -m findex.ranking_checks
"""
from pathlib import Path

from findex.index import build_index
from findex.models import Index, Posting
from findex.ranking import BM25, TfIdf, search


def rarest_word(index: Index) -> str:
    words = [t for t in index if t.isalpha() and len(t) > 2]
    return min(words, key=lambda t: (index.df(t), t))


def check_rare_beats_common(index: Index) -> None:
    print("=== Перевірка 1: рідкісний термін вище за частий ===")
    common = max(index, key=lambda t: (index.df(t), sum(p.freq for p in index[t])))
    rare = rarest_word(index)
    print(f"частий термін: {common!r} (є в {index.df(common)} документах)")
    print(f"рідкісний термін: {rare!r} (є в {index.df(rare)} документах)")

    results = search(index, f"{common} {rare}", k=3)
    for r in results:
        print(f"  {r}")
    has_rare = any(p.doc_id == results[0].doc_id for p in index[rare])
    print("ОК: на першому місці документ із рідкісним терміном" if has_rare else "УВАГА: перевір корпус")
    print()


def check_saturation(index: Index) -> None:
    print("=== Перевірка 2: 20-те повторення майже нічого не додає ===")
    rare = rarest_word(index)
    base = index[rare][0]
    bm25, tfidf = BM25(), TfIdf()

    def contribution(scorer, tf: int) -> float:
        return scorer(rare, Posting(base.doc_id, tf), index)

    print(f"термін {rare!r}, документ [{base.doc_id}]; внесок у рахунок залежно від tf:")
    print(f"  {'tf':>3} {'BM25':>8} {'TF-IDF':>8}")
    for tf in (1, 2, 5, 10, 20, 50):
        print(f"  {tf:>3} {contribution(bm25, tf):8.3f} {contribution(tfidf, tf):8.3f}")

    for name, scorer in (("BM25", bm25), ("TF-IDF", tfidf)):
        first = contribution(scorer, 1)
        gain_20th = contribution(scorer, 20) - contribution(scorer, 19)
        print(f"{name}: 20-те повторення додає {gain_20th:.4f} = {gain_20th / first:.1%} від першого входження")
    print()


def check_short_beats_long(index: Index) -> None:
    print("=== Перевірка 3: короткий документ вище за дуже довгий ===")
    bm25 = BM25()
    best = None  # (ratio, term, short_posting, long_posting)
    for term in index:
        once = [p for p in index[term] if p.freq == 1]
        if len(once) < 2:
            continue
        short = min(once, key=lambda p: index.doc_length(p.doc_id))
        long_ = max(once, key=lambda p: index.doc_length(p.doc_id))
        ratio = index.doc_length(long_.doc_id) / index.doc_length(short.doc_id)
        if best is None or ratio > best[0]:
            best = (ratio, term, short, long_)

    if best is None:
        print("У корпусі немає терміна, що по одному разу зустрічається у двох документах.")
        return
    _ratio, term, short, long_ = best
    for label, p in (("короткий", short), ("довгий", long_)):
        meta = index.doc(p.doc_id)
        print(f"  {label:9} {meta.path.stem[:40]:40} довжина {meta.length:>7}  BM25={bm25(term, p, index):.3f}")
    ok = bm25(term, short, index) > bm25(term, long_, index)
    print(f"термін {term!r}, в обох по одному входженню. " + ("ОК: короткий вище" if ok else "УВАГА"))
    print()


if __name__ == "__main__":
    index = build_index(Path("data/raw"))
    print(index, end="\n\n")
    check_rare_beats_common(index)
    check_saturation(index)
    check_short_beats_long(index)