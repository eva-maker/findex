"""Посилання, заголовок і видимий текст зі сторінки (html.parser зі стандартної бібліотеки)."""

from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

from findex.crawler.urls import canonicalize

_SKIPPED_TAGS = frozenset({"script", "style", "noscript", "template", "head"})
# Блокові теги розділяють слова пробілом: «<h1>Page 0</h1><p>unique» не має злипнутись у «0unique».
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "nav", "section", "article",
     "header", "footer", "main", "aside", "blockquote", "pre", "hr", "h1", "h2", "h3", "h4", "h5",
     "h6", "dt", "dd", "form", "figure", "figcaption"}
)  # fmt: skip


@dataclass(slots=True)
class ParsedPage:
    title: str
    text: str
    links: list[str] = field(default_factory=list[str])


class _Extractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []
        self._seen_links: set[str] = set()
        self._text: list[str] = []
        self._title: list[str] = []
        self._skip_depth = 0  # скільки відкритих script/style/... зараз
        self._in_title = False
        self._base_taken = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._text.append(" ")
        if tag == "title":
            self._in_title = True
        elif tag in _SKIPPED_TAGS and tag != "head":
            self._skip_depth += 1
        elif tag == "base" and not self._base_taken:
            href = dict(attrs).get("href")
            if href:
                self.base_url = urljoin(self.base_url, href)  # сирий URL: слеш наприкінці важливий
                self._base_taken = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                link = canonicalize(href.strip(), self.base_url)
                if link is not None and link not in self._seen_links:
                    self._seen_links.add(link)
                    self.links.append(link)

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self._text.append(" ")
        if tag == "title":
            self._in_title = False
        elif tag in _SKIPPED_TAGS and tag != "head" and self._skip_depth > 0:
            self._skip_depth -= 1

    @property
    def title(self) -> str:
        return " ".join("".join(self._title).split())

    @property
    def text(self) -> str:
        return " ".join("".join(self._text).split())

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif self._skip_depth == 0:
            self._text.append(data)


def parse_html(html: str, base_url: str) -> ParsedPage:
    """Вийняти заголовок, текст і канонічні посилання (без дублів, у порядку появи)."""
    extractor = _Extractor(base_url)
    extractor.feed(html)
    extractor.close()
    return ParsedPage(title=extractor.title, text=extractor.text, links=extractor.links)
