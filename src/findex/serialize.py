"""
pickle зручний (серіалізує довільні Python-об'єкти, включно з нашими
dataclasses, без жодного бойлерплейту), але небезпечний для завантаження
з недовіреного джерела: unpickle може виконати довільний код, вбудований
у файл (Python викликає __reduce__/__setstate__ під час load, і спеціально
сформований файл змусить цей виклик зробити що завгодно, включно з
os.system(...)). Завантажуй pickle-файли тільки ті, що збудував сам, або
чиєму джерелу повністю довіряєш.

json — безпечніша, читабельна альтернатива для порівняння: може
представляти лише прості дані (dict/list/str/int/float/bool/None), тому
завантаження json не може виконати код — але наші Posting/DocMeta
dataclasses не серіалізуються в json напряму, тому конвертуємо їх у прості
dict/list при збереженні й відновлюємо dataclasses при завантаженні.
"""
import json
import pickle
from contextlib import contextmanager
from pathlib import Path

from findex.models import DocMeta, Index, Posting

from findex.timing import timed


def save_pickle(index: Index, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)

@timed
def load_pickle(path: Path) -> Index:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(index: Index, path: Path) -> None:
    def plain(p: Posting) -> list:
        row = [p.doc_id, p.freq]
        if p.positions:
            row.append(list(p.positions))      # третім елементом — позиції, якщо є
        return row

    plain_postings = {
        term: [plain(p) for p in plist]
        for term, plist in index.items()
    }
    plain_doc_meta = {
        str(m.doc_id): {"doc_id": m.doc_id, "path": str(m.path), "length": m.length}
        for m in index.docs()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"postings": plain_postings, "doc_meta": plain_doc_meta}, f)


def load_json(path: Path) -> Index:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    postings = {
        term: [Posting(d, fr, tuple(rest[0]) if rest else ()) for d, fr, *rest in plist]
        for term, plist in data["postings"].items()
    }
    doc_meta = {
        int(k): DocMeta(doc_id=m["doc_id"], path=Path(m["path"]), length=m["length"])
        for k, m in data["doc_meta"].items()
    }
    return Index(postings, doc_meta)


def save_index(index: Index, path: Path) -> None:
    if path.suffix == ".json":
        save_json(index, path)
    else:
        save_pickle(index, path)


def load_index(path: Path) -> Index:
    if path.suffix == ".json":
        return load_json(path)
    return load_pickle(path)


@contextmanager
def open_index(path: Path):
    """with open_index(p) as ix: ... — індекс гарантовано закривається при виході,
    навіть якщо всередині with стався виняток або return."""
    index = load_index(path)
    try:
        yield index            # все, що до yield — це __enter__
    finally:
        index.close()          # все, що після — це __exit__
