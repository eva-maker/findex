import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Логує, скільки тривала функція (навіть якщо вона кинула виняток).

    [**P, R] — generic із синтаксисом PEP 695: P — це «список параметрів»
    обгорнутої функції, R — її результат. Завдяки цьому pyright знає, що
    після @timed сигнатура функції не змінилась.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            name = getattr(fn, "__name__", repr(fn))
            log.debug("%s took %.1f ms", name, (time.perf_counter() - t0) * 1000)

    return wrapper
