import argparse
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

from findex.corpus import iter_documents_with_meta
from findex.models import DocMeta, Index, Posting
from findex.serialize import save_index
from findex.tokens import tokenize


def build_index(corpus_dir: Path, positions: bool = False) -> Index:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents_with_meta(corpus_dir)):
        term_freq_in_doc: defaultdict[str, int] = defaultdict(int)
        term_positions: defaultdict[str, list[int]] = defaultdict(list)
        token_count = 0
        for token in tokenize(doc.text):
            term_freq_in_doc[token] += 1
            if positions:
                term_positions[token].append(token_count)   # номер токена в документі
            token_count += 1

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=doc.path, length=token_count)

        for term, freq in term_freq_in_doc.items():
            where = tuple(term_positions[term]) if positions else ()
            postings[term].append(Posting(doc_id=doc_id, freq=freq, positions=where))

    return Index(postings, doc_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an inverted index from a corpus.")
    parser.add_argument("corpus_dir", type=Path, help="Directory with .txt documents")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Output index file")
    parser.add_argument("--positions", action="store_true", help="Store token positions (needed for phrase queries)")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()

    index = build_index(args.corpus_dir, positions=args.positions)
    save_index(index, args.out)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(index)
    print(f"Saved to {args.out}")
    print(f"Time:        {elapsed:.4f}s")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()