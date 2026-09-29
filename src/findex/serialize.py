"""
pickle зручний (серіалізує довільні Python-об'єкти, включно з нашими
dataclasses, без жодного бойлерплейту), але НЕБЕЗПЕЧНИЙ для завантаження
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
from collections import defaultdict
from pathlib import Path

from findex.models import DocMeta, Posting


def save_pickle(postings: dict, doc_meta: dict, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"postings": postings, "doc_meta": doc_meta}, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["postings"], data["doc_meta"]


def save_json(postings: dict, doc_meta: dict, path: Path) -> None:
    plain_postings = {
        term: [[p.doc_id, p.freq] for p in plist]
        for term, plist in postings.items()
    }
    plain_doc_meta = {
        str(doc_id): {"doc_id": m.doc_id, "path": str(m.path), "length": m.length}
        for doc_id, m in doc_meta.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"postings": plain_postings, "doc_meta": plain_doc_meta}, f)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    postings = defaultdict(list)
    for term, plist in data["postings"].items():
        postings[term] = [Posting(doc_id=doc_id, freq=freq) for doc_id, freq in plist]

    doc_meta = {
        int(doc_id): DocMeta(doc_id=m["doc_id"], path=Path(m["path"]), length=m["length"])
        for doc_id, m in data["doc_meta"].items()
    }
    return postings, doc_meta