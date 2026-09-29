from collections import defaultdict
from pathlib import Path

from findex.corpus import iter_documents_with_meta
from findex.models import DocMeta, Posting
from findex.tokens import tokenize

import argparse
import time
import tracemalloc

from findex.serialize import save_json, save_pickle



def build_index(corpus_dir: Path):
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents_with_meta(corpus_dir)):
        term_freq_in_doc: defaultdict[str, int] = defaultdict(int)
        token_count = 0
        for token in tokenize(doc.text):
            term_freq_in_doc[token] += 1
            token_count += 1

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=doc.path, length=token_count)

        for term, freq in term_freq_in_doc.items():
            postings[term].append(Posting(doc_id=doc_id, freq=freq))

    return postings, doc_meta


def _save_index(postings: dict, doc_meta: dict, out_path: Path) -> None:
    if out_path.suffix == ".json":
        save_json(postings, doc_meta, out_path)
    else:
        save_pickle(postings, doc_meta, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an inverted index from a corpus.")
    parser.add_argument("corpus_dir", type=Path, help="Directory with .txt documents")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Output index file")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()

    postings, doc_meta = build_index(args.corpus_dir)
    _save_index(postings, doc_meta, args.out)

    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Indexed {len(doc_meta)} documents, {len(postings)} unique terms")
    print(f"Saved to {args.out}")
    print(f"Time:        {elapsed:.4f}s")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()