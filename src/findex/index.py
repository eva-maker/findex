"""Побудова індексу: послідовна (Лаба 4) і паралельна (Лаба 5).

Паралельна збірка ділиться на дві частини:

  * ``build_partial`` — індексує свій шматок корпусу й повертає ``PartialIndex``.
    Це функція рівня модуля, тому її можна віддати іншому процесу (pickle
    передає лише її ім'я, а не код). У воркер їдуть ШЛЯХИ до файлів, а не вже
    прочитаний текст: файли воркер читає сам.
  * ``merge`` — склеює часткові індекси в один ``Index``. Працює в одному
    процесі, тож це та частина роботи, яка не прискорюється воркерами.

Щоб часткові індекси можна було склеїти простою конкатенацією, кожному шматку
наперед віддається свій діапазон ``doc_id`` (див. ``plan_chunks``).
"""

import logging
import multiprocessing
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, get_args

from findex.corpus import Document, iter_documents_with_meta, load_document
from findex.models import DocMeta, Index, Posting
from findex.timing import timed
from findex.tokens import tokenize

log = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]
EXECUTOR_NAMES: tuple[str, ...] = get_args(ExecutorName)


@dataclass(slots=True)
class PartialIndex:
    """Індекс одного шматка корпусу. Його doc_id — діапазон, виданий наперед.

    ``worker_pid`` і ``worker_cpu`` — службові поля для вимірювань: який процес
    це збудував і скільки процесорного часу він витратив за все життя (щоб у
    бенчмарку скласти CPU time усіх воркерів).
    """

    postings: dict[str, list[Posting]]
    doc_meta: dict[int, DocMeta]
    worker_pid: int = 0
    worker_cpu: float = 0.0


@dataclass(frozen=True, slots=True)
class Chunk:
    """Шматок корпусу: шляхи файлів і doc_id першого з них."""

    first_doc_id: int
    paths: tuple[Path, ...]


def partial_from_documents(
    docs: Iterable[Document], first_doc_id: int = 0, positions: bool = False
) -> PartialIndex:
    """Чиста логіка: документи -> часткові постінги. Читання файлів тут нема.

    doc_id рахується від ``first_doc_id``. Усі постінги кожного терміна йдуть у
    порядку документів, тобто відсортовані за doc_id без явного sort.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(docs, start=first_doc_id):
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

    return PartialIndex(dict(postings), doc_meta)


def build_partial(
    paths: Sequence[Path], first_doc_id: int = 0, positions: bool = False
) -> PartialIndex:
    """Проіндексувати файли ``paths``, давши їм doc_id ``first_doc_id``, ``first_doc_id + 1``, ...

    Функція рівня модуля і приймає лише прості дані (шляхи, числа), тому її можна
    відправити в інший процес. Документи читаються по одному, не всі одразу.
    """
    part = partial_from_documents(
        (load_document(path) for path in paths), first_doc_id=first_doc_id, positions=positions
    )
    part.worker_pid = os.getpid()
    # process_time — CPU цього процесу від його старту (разом з імпортами):
    # так у бенчмарку видно й ціну запуску воркера.
    part.worker_cpu = time.process_time()
    return part


def merge(parts: Iterable[PartialIndex]) -> Index:
    """Склеїти часткові індекси в один.

    Діапазони doc_id не повинні перетинатися: шматки впорядковуються за першим
    doc_id, і тоді списки постінгів кожного терміна просто дописуються один за
    одним, лишаючись відсортованими. Перетин діапазонів — ValueError.

    Списки постінгів з перших частин переходять у результат без копіювання,
    тому часткові індекси після merge використовувати не можна.
    """
    ordered = sorted((p for p in parts if p.doc_meta), key=lambda p: min(p.doc_meta))
    postings: dict[str, list[Posting]] = {}
    doc_meta: dict[int, DocMeta] = {}
    last_doc_id = -1

    for part in ordered:
        first = min(part.doc_meta)
        if first <= last_doc_id:
            raise ValueError(
                f"діапазони doc_id перетинаються: частина починається з {first}, "
                f"а попередня дійшла до {last_doc_id}"
            )
        last_doc_id = max(part.doc_meta)
        doc_meta.update(part.doc_meta)
        for term, plist in part.postings.items():
            existing = postings.get(term)
            if existing is None:
                postings[term] = plist
            else:
                existing.extend(plist)

    index = Index(postings, doc_meta)
    log.info("merge: %d документів, %d термінів", index.num_docs, len(index))
    return index


def index_documents(docs: Iterable[Document], positions: bool = False) -> Index:
    """Чиста логіка побудови індексу: бере документи, повертає Index.

    Тут немає читання файлів і друку, тому її легко тестувати на маленьких
    рядках у пам'яті («hoist your I/O»). Приймаємо Iterable, щоб документи
    могли надходити стрімом, по одному.
    """
    part = partial_from_documents(docs, first_doc_id=0, positions=positions)
    index = Index(part.postings, part.doc_meta)
    log.info("проіндексовано %d документів, %d термінів", index.num_docs, len(index))
    return index


@timed
def build_index(corpus_dir: Path, positions: bool = False) -> Index:
    """Тонка обгортка: читає каталог з диска й віддає документи в index_documents."""
    return index_documents(iter_documents_with_meta(corpus_dir), positions=positions)


def plan_chunks(
    paths: Sequence[Path], parts: int, sizes: Sequence[int] | None = None
) -> list[Chunk]:
    """Розділити файли на ``parts`` суцільних шматків приблизно однакового розміру в байтах.

    Шматок суцільний (файли йдуть підряд), тож діапазон його doc_id — просто
    номери цих файлів у відсортованому списку. Розмір беремо зі ``stat``: вміст
    файлів тут не читається. Шматків не більше, ніж файлів, і жоден не порожній.
    """
    count = len(paths)
    if count == 0:
        return []
    parts = max(1, min(parts, count))
    if sizes is None:
        sizes = [path.stat().st_size for path in paths]
    total = sum(sizes) or count

    chunks: list[Chunk] = []
    start = 0
    acc = 0
    for i, size in enumerate(sizes):
        acc += size
        done = len(chunks)
        if done == parts - 1:
            break
        items_left = count - i - 1
        chunks_left = parts - done - 1
        reached_share = acc >= total * (done + 1) / parts
        # Різати можна, лише якщо лишається досить файлів для решти шматків;
        # а якщо файлів рівно стільки, скільки шматків, різати треба обов'язково.
        if items_left == chunks_left or (reached_share and items_left > chunks_left):
            chunks.append(Chunk(first_doc_id=start, paths=tuple(paths[start : i + 1])))
            start = i + 1
    chunks.append(Chunk(first_doc_id=start, paths=tuple(paths[start:])))
    return chunks


def build_chunk(chunk: Chunk, positions: bool) -> PartialIndex:
    """Робота одного воркера. Окрема функція рівня модуля, щоб її можна було pickle-ити."""
    return build_partial(chunk.paths, chunk.first_doc_id, positions)


def _make_pool(kind: Literal["threads", "processes"], workers: int) -> Executor:
    if kind == "threads":
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
    # Явно spawn: на Windows і macOS він і так типовий, а на Linux робить поведінку
    # однаковою скрізь. Воркер стартує з чистого інтерпретатора й імпортує модуль
    # заново, тому вхідна точка програми має бути захищена `if __name__ == "__main__"`.
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def map_in_executor[A, R](
    fn: Callable[[A], R],
    items: Sequence[A],
    *,
    workers: int,
    executor: ExecutorName,
    on_done: Callable[[int], None] | None = None,
) -> list[R]:
    """Виконати ``fn`` для кожного елемента; результати повертаються в порядку ``items``.

    Три бекенди з однаковою поведінкою: звичайний цикл, потоки, процеси. Якщо
    хоч одне завдання впало, виняток піднімається одразу (для процесів з
    віддаленим стеком воркера в ланцюжку причин), невиконані завдання
    скасовуються, і програма не зависає.
    """
    if executor == "serial":
        results: list[R] = []
        for i, item in enumerate(items):
            results.append(fn(item))
            if on_done is not None:
                on_done(i)
        return results

    with _make_pool(executor, workers) as pool:
        futures: list[Future[R]] = [pool.submit(fn, item) for item in items]
        position = {future: i for i, future in enumerate(futures)}
        try:
            for future in as_completed(futures):
                future.result()  # якщо воркер упав, виняток вилітає тут
                if on_done is not None:
                    on_done(position[future])
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [future.result() for future in futures]


def build_partials(
    plan: Sequence[Chunk],
    *,
    positions: bool = False,
    workers: int = 1,
    executor: ExecutorName = "serial",
    on_chunk_done: Callable[[Chunk], None] | None = None,
) -> list[PartialIndex]:
    """Фаза 1: часткові індекси для кожного шматка плану (паралельна частина роботи)."""

    def done(i: int) -> None:
        if on_chunk_done is not None:
            on_chunk_done(plan[i])

    return map_in_executor(
        partial(build_chunk, positions=positions),
        plan,
        workers=workers,
        executor=executor,
        on_done=done,
    )


def worker_cpu_seconds(parts: Iterable[PartialIndex]) -> float:
    """Сумарний CPU time воркерів-ПРОЦЕСІВ (без поточного процесу).

    Кожен воркер повідомляє свій накопичений process_time після кожного шматка;
    для кожного процесу беремо найбільше значення (останнє).
    """
    me = os.getpid()
    latest: dict[int, float] = {}
    for part in parts:
        if part.worker_pid != me:
            latest[part.worker_pid] = max(latest.get(part.worker_pid, 0.0), part.worker_cpu)
    return sum(latest.values())


def build_index_parallel(
    paths: Sequence[Path],
    *,
    positions: bool = False,
    workers: int = 1,
    executor: ExecutorName = "serial",
    chunks: int | None = None,
    on_chunk_done: Callable[[Chunk], None] | None = None,
) -> Index:
    """plan_chunks -> build_partials (паралельно) -> merge (послідовно).

    Шматків за замовчуванням стільки, скільки воркерів (для ``serial`` — один).
    """
    if workers < 1:
        raise ValueError("workers має бути не менше 1")
    wanted = chunks if chunks is not None else (1 if executor == "serial" else workers)
    plan = plan_chunks(paths, wanted)
    log.info(
        "executor=%s workers=%d шматків=%d файлів=%d", executor, workers, len(plan), len(paths)
    )
    parts = build_partials(
        plan, positions=positions, workers=workers, executor=executor, on_chunk_done=on_chunk_done
    )
    return merge(parts)
