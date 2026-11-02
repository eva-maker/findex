"""Графік прискорення з результатів parallel_benchmark.

    uv run --group bench python scripts/plot_speedup.py docs/bench_gil.json -o docs/speedup.png
    uv run --group bench python scripts/plot_speedup.py docs/bench_gil.json docs/bench_nogil.json

Кожен JSON — один запуск `python -m findex.parallel_benchmark` (одна збірка Python).
Якщо JSON-ів два (звичайний Python і 3.13t), на одному графіку будуть потоки з GIL
і без нього. Пряма «ідеально» — лінійне зростання: N воркерів дають N-кратне прискорення.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без вікна: лише зберегти PNG

import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "processes": "#1f6fb2",
    "threads": "#d9822b",
    "threads, без GIL": "#2e9e5b",
    "processes (без GIL)": "#5b8fc9",
    "ідеально": "#8a8a8a",
}


def load(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["meta"], data["rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path, help="JSON з parallel_benchmark")
    parser.add_argument("-o", "--out", type=Path, default=Path("docs/speedup.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    max_workers = 1
    meta_first = None
    for path in args.results:
        meta, rows = load(path)
        meta_first = meta_first or meta
        for executor in ("processes", "threads"):
            points = sorted((r["workers"], r["speedup"]) for r in rows if r["executor"] == executor)
            if not points:
                continue
            label = executor
            if executor == "threads" and meta["gil"] is False:
                label = "threads, без GIL"
            if executor == "processes" and meta["gil"] is False:
                label = "processes (без GIL)"
            xs, ys = zip(*points, strict=True)
            max_workers = max(max_workers, *xs)
            ax.plot(xs, ys, marker="o", linewidth=2, label=label, color=COLORS.get(label, "#555"))

    ax.plot([1, max_workers], [1, max_workers], "--", color=COLORS["ідеально"], label="ідеально")
    ax.axhline(1, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_xlabel("Воркерів")
    ax.set_ylabel("Прискорення (wall serial / wall)")
    ax.set_xscale("log", base=2)
    ticks = sorted({r["workers"] for p in args.results for r in load(p)[1]})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if meta_first:
        ax.set_title(
            f"Індексація корпусу: {meta_first['cpu_model']}, {meta_first['logical_cores']} логічних ядер",
            fontsize=10,
        )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Збережено {args.out}")


if __name__ == "__main__":
    main()
