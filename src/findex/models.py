import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    freq: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: Path
    length: int  # кількість токенів у документі


class Index(Mapping):
    def __init__(self, postings: dict[str, list[Posting]], doc_meta: dict[int, DocMeta]):
        self._postings = dict(postings)
        self._doc_meta = dict(doc_meta)


    def __len__(self) -> int:
        return len(self._postings)          # len(index) = кількість термінів

    def __contains__(self, term: object) -> bool:
        return term in self._postings       # "python" in index

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]         # index["python"]; немає терміна -> KeyError

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)         # for term in index

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"


    @property
    def num_docs(self) -> int:
        return len(self._doc_meta)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_meta:
            return 0.0
        return sum(m.length for m in self._doc_meta.values()) / self.num_docs


    def doc_length(self, doc_id: int) -> int:
        return self._doc_meta[doc_id].length

    def df(self, term: str) -> int:
        """Document frequency: у скількох документах є цей термін."""
        return len(self._postings.get(term, ()))

    def doc(self, doc_id: int) -> DocMeta:
        return self._doc_meta[doc_id]

    def docs(self) -> Iterator[DocMeta]:
        return iter(self._doc_meta.values())

    def close(self) -> None:
        self._postings = {}
        self._doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)