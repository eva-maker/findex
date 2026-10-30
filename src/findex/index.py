import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from findex.corpus import Document, iter_documents_with_meta
from findex.models import DocMeta, Index, Posting
from findex.timing import timed
from findex.tokens import tokenize

log = logging.getLogger(__name__)


def index_documents(docs: Iterable[Document], positions: bool = False) -> Index:
    """Чиста логіка побудови індексу: бере документи, повертає Index.

    Тут немає читання файлів і друку, тому її легко тестувати на маленьких
    рядках у пам'яті («hoist your I/O»). Приймаємо Iterable, щоб документи
    могли надходити стрімом, по одному.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs):
        term_freq_in_doc: defaultdict[str, int] = defaultdict(int)
        term_positions: defaultdict[str, list[int]] = defaultdict(list)
        token_count = 0
        for token in tokenize(doc.text):
            term_freq_in_doc[token] += 1
            if positions:
                term_positions[token].append(token_count)  # номер токена в документі
            token_count += 1

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=doc.path, length=token_count)
        log.debug("doc %d %s: %d токенів", doc_id, doc.path.name, token_count)

        for term, freq in term_freq_in_doc.items():
            where = tuple(term_positions[term]) if positions else ()
            postings[term].append(Posting(doc_id=doc_id, freq=freq, positions=where))

    index = Index(postings, doc_meta)
    log.info("проіндексовано %d документів, %d термінів", index.num_docs, len(index))
    return index


@timed
def build_index(corpus_dir: Path, positions: bool = False) -> Index:
    """Тонка обгортка: читає каталог з диска й віддає документи в index_documents."""
    return index_documents(iter_documents_with_meta(corpus_dir), positions=positions)
