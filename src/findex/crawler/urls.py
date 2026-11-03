"""Канонізація адрес і перевірка домену.

Дві адреси, що ведуть на ту саму сторінку, мають мати однаковий запис, інакше множина
``seen`` не вбереже від повторного завантаження. Канонізуємо так:

  * схема і хост у нижньому регістрі, порт за замовчуванням (80/443) прибираємо;
  * фрагмент після ``#`` відкидаємо (сервер його не бачить);
  * порожній шлях -> ``/``, слеш наприкінці прибираємо (крім кореня);
  * параметри query сортуємо за назвою.
"""

from collections.abc import Collection
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_DEFAULT_PORTS = {"http": 80, "https": 443}

# Розширення, за якими видно, що це не HTML: такі адреси навіть не просимо.
_SKIP_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".mp3", ".mp4", ".avi", ".css", ".js", ".json", ".xml", ".woff", ".woff2",
)  # fmt: skip


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Привести адресу до канонічного вигляду; None, якщо це не http(s)-адреса.

    ``base`` потрібен для відносних посилань (``../a.html``).
    """
    try:
        parts = urlsplit(urljoin(base, url) if base is not None else url)
        host = parts.hostname
        port = parts.port
    except ValueError:  # напр. "http://[bad" або порт не число
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None

    netloc = f"[{host}]" if ":" in host else host  # IPv6 повертаємо в квадратних дужках
    if port is not None and port != _DEFAULT_PORTS[scheme]:
        netloc = f"{netloc}:{port}"

    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """``host[:port]`` канонічної адреси: ключ для ліміту на хост і кешу robots.txt."""
    return urlsplit(url).netloc


def looks_like_page(url: str) -> bool:
    """False для адрес, що явно ведуть на картинку, архів чи скрипт."""
    return not urlsplit(url).path.lower().endswith(_SKIP_SUFFIXES)


def in_allowed_domains(url: str, domains: Collection[str], *, subdomains: bool = True) -> bool:
    """Хост збігається з дозволеним доменом (і, якщо ``subdomains``, з його піддоменами)."""
    host = urlsplit(url).hostname or ""
    return any(host == d or (subdomains and host.endswith("." + d)) for d in domains)
