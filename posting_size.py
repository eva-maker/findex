"""Скільки байтів займає один постінг у трьох варіантах + як росте словник.

"""

import gc
import sys
import tracemalloc
from array import array
from collections.abc import Callable
from dataclasses import dataclass

N = 100_000  # кількість постінгів для виміру


@dataclass
class PostingPlain:  # звичайний dataclass: без slots, без frozen
    doc_id: int
    freq: int


@dataclass(frozen=True, slots=True)
class PostingSlots:  # як Posting у Лабі 2: два поля, slots, frozen
    doc_id: int
    freq: int


def bytes_per_item(make_items: Callable[[], object]) -> float:
    """Міряє через tracemalloc, скільки пам'яті займає N штук."""
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    items = make_items()  # для збереження результату
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del items
    return (after - before) / N


# як у справжньому індексі: doc_id від 0 до 19, freq невеликі.
# Малі цілі (до 256) Python не створює щоразу, а бере готові 
plain = bytes_per_item(lambda: [PostingPlain(i % 20, i % 5 + 1) for i in range(N)])
slots = bytes_per_item(lambda: [PostingSlots(i % 20, i % 5 + 1) for i in range(N)])


def make_array():
    a = array("I")
    for i in range(N):
        a.append(i % 20)  # doc_id
        a.append(i % 5 + 1)  # freq
    return a


arr = bytes_per_item(make_array)

print("Байтів на один постінг (разом зі списком, де він лежить):")
print(f"  plain dataclass : {plain:6.1f}")
print(f"  slots dataclass : {slots:6.1f}")
print(f"  array('I')      : {arr:6.1f}  (пара doc_id + freq)")
print(f"  plain / array   : {plain / arr:5.1f} разів")

print("\nЯк росте словник (sys.getsizeof), ключів -> байт:")
d: dict[int, int] = {}
last = sys.getsizeof(d)
print(f"  {len(d):>3} -> {last}")
for i in range(40):
    d[i] = i
    now = sys.getsizeof(d)
    if now != last:
        print(f"  {len(d):>3} -> {now}")
        last = now
