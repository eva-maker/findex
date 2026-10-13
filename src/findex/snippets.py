from collections.abc import Iterable

from findex.tokens import normalize_text, token_spans


def _find_matches(text: str, terms: Iterable[str]) -> list[tuple[str, int, int]]:
    wanted = set(terms)
    return [(t, s, e) for t, s, e in token_spans(text) if t in wanted]


def _best_center(matches: list[tuple[str, int, int]], width: int) -> int:
    """Початок збігу, навколо якого вікно містить найбільше РІЗНИХ термінів."""
    best_start, best_score = matches[0][1], -1
    for _t, start, _e in matches:
        score = len({t for t, s, _ in matches if start <= s <= start + width})
        if score > best_score:
            best_start, best_score = start, score
    return best_start


def make_snippet(text: str, terms: Iterable[str], width: int = 80, mark: str = "**") -> str:
    norm = normalize_text(text)
    matches = _find_matches(norm, terms)
    if not matches:
        return ""
    center = _best_center(matches, width)
    lo = max(0, center - width)
    hi = min(len(norm), center + width)
    while 0 < lo < center and not norm[lo - 1].isspace():  # не різати слова
        lo += 1
    while center < hi < len(norm) and not norm[hi].isspace():
        hi -= 1
    pieces, pos = [], lo
    for _t, s, e in matches:
        if s >= lo and e <= hi:
            pieces += [norm[pos:s], mark, norm[s:e], mark]
            pos = e
    pieces.append(norm[pos:hi])
    body = " ".join("".join(pieces).split())
    return ("… " if lo > 0 else "") + body + (" …" if hi < len(norm) else "")