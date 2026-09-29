from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents_with_meta
from findex.tokens import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    freq: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: Path
    length: int  # кількість токенів у документі


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


if __name__ == "__main__":
    postings, doc_meta = build_index(Path("data/raw"))
    print(f"Unique terms: {len(postings)}")
    print(f"Documents: {len(doc_meta)}")

    the_postings = postings["the"]
    total_the = sum(p.freq for p in the_postings)
    print(f"'the' зустрічається в {len(the_postings)} документах, разом {total_the} разів")

    for term, plist in list(postings.items())[:5]:
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids), f"не відсортовано: {term}"
    print("перевірка сортування: ок (на вибірці)")