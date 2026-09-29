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