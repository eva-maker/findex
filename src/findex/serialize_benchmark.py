import time
from pathlib import Path

from findex.index import build_index
from findex.serialize import load_json, load_pickle, save_json, save_pickle


def bench_format(name: str, save_fn, load_fn, postings, doc_meta, path: Path):
    start = time.perf_counter()
    save_fn(postings, doc_meta, path)
    save_time = time.perf_counter() - start

    size_kb = path.stat().st_size / 1024

    start = time.perf_counter()
    load_fn(path)
    load_time = time.perf_counter() - start

    print(f"{name:<10} {size_kb:>10.1f} KB   save: {save_time*1000:>7.2f} ms   load: {load_time*1000:>7.2f} ms")


if __name__ == "__main__":
    postings, doc_meta = build_index(Path("data/raw"))

    print(f"{'Формат':<10} {'Розмір':>13}   {'Save':>12}   {'Load':>12}")
    bench_format("pickle", save_pickle, load_pickle, postings, doc_meta, Path("index.pkl"))
    bench_format("json", save_json, load_json, postings, doc_meta, Path("index.json"))