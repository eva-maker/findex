import functools
import logging
import time

log = logging.getLogger("findex")


def timed(fn):
    """Логує, скільки тривала функція (навіть якщо вона кинула виняток)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.1f ms", fn.__name__, (time.perf_counter() - t0) * 1000)

    return wrapper