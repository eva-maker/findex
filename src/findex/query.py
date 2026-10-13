"""Мова запитів: лексер, рекурсивний спуск, дерево вузлів.

Синтаксис:  python AND (async OR await) NOT java "event loop"
  - слова поспіль = неявний AND
  - AND / OR / NOT — лише ВЕЛИКИМИ літерами (щоб слово "or" можна було шукати)
  - дужки змінюють пріоритет, "в лапках" = фраза

Граматика (пріоритет зростає знизу вгору):
    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr (['AND'] not_expr)*
    not_expr -> 'NOT' not_expr | atom
    atom     -> TERM | PHRASE | '(' or_expr ')'
"""
import re
from dataclasses import dataclass

from findex.models import Index
from findex.tokens import tokenize


class QueryError(ValueError):
    """Запит не вдалося розібрати або виконати."""


# вузли дерева

class Node:
    """Базовий клас: дає всім вузлам оператори &, |, ~ та спільний інтерфейс."""

    def __and__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return And(self, other)

    def __or__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> list[str]:
        """Терміни, що додають бали в рейтинг (все, крім тих, що під NOT)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.get(self.text, ())}

    def positive_terms(self) -> list[str]:
        return [self.text]


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise QueryError("фразовий пошук потребує індексу з позиціями: findex.index --positions")

        # для кожного слова фрази: doc_id -> позиції в цьому документі
        per_term = []
        for term in self.terms:
            if term not in index:
                return set()
            per_term.append({p.doc_id: p.positions for p in index[term]})

        result = set()
        for doc_id in set.intersection(*(set(d) for d in per_term)):
            starts = set(per_term[0][doc_id])            # де може починатися фраза
            for offset, postings in enumerate(per_term[1:], start=1):
                later = set(postings[doc_id])
                starts = {s for s in starts if s + offset in later}
                if not starts:
                    break
            if starts:
                result.add(doc_id)
        return result

    def positive_terms(self) -> list[str]:
        return list(self.terms)


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Not(Node):
    operand: Node

    def evaluate(self, index: Index) -> set[int]:
        return index.doc_ids() - self.operand.evaluate(index)

    def positive_terms(self) -> list[str]:
        return []


# лексер

_LEXEME_RE = re.compile(
    r'\s*(?:(?P<lpar>\()|(?P<rpar>\))|"(?P<phrase>[^"]*)"|(?P<word>[^\s()"]+))'
)
_OPERATORS = ("AND", "OR", "NOT")


def _lex(text: str) -> list[tuple[str, object]]:
    """Рядок -> список токенів (вид, значення). Слова проходять через tokenize()
    з Lab 1, тож запит нормалізується так само, як документи."""
    text = text.strip()
    tokens: list[tuple[str, object]] = []
    pos = 0
    while pos < len(text):
        m = _LEXEME_RE.match(text, pos)
        if m is None:
            raise QueryError(f"не вдалося розібрати запит біля {text[pos:pos + 12]!r} (незакриті лапки?)")
        pos = m.end()

        if m["lpar"]:
            tokens.append(("LPAR", None))
        elif m["rpar"]:
            tokens.append(("RPAR", None))
        elif m["phrase"] is not None:
            words = tuple(tokenize(m["phrase"]))
            if not words:
                raise QueryError("порожня фраза в лапках")
            tokens.append(("PHRASE", words))
        else:
            raw = m["word"]
            if raw in _OPERATORS:
                tokens.append((raw, None))
            else:
                for word in tokenize(raw):       # "foo.bar" -> foo, bar
                    tokens.append(("TERM", word))
    return tokens


# парсер

class _Parser:
    def __init__(self, tokens: list[tuple[str, object]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def advance(self) -> tuple[str, object]:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self) -> Node:
        if not self.tokens:
            raise QueryError("порожній запит")
        node = self.or_expr()
        if self.pos != len(self.tokens):
            raise QueryError(f"зайвий елемент у запиті: {self.peek()}")
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.advance()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while True:
            kind = self.peek()
            if kind == "AND":
                self.advance()
                node = And(node, self.not_expr())
            elif kind in ("TERM", "PHRASE", "LPAR", "NOT"):   # неявний AND
                node = And(node, self.not_expr())
            else:
                return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.advance()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind = self.peek()
        if kind is None:
            raise QueryError("запит закінчився раніше, ніж очікувалось")
        if kind == "TERM":
            return Term(self.advance()[1])
        if kind == "PHRASE":
            words = self.advance()[1]
            return Term(words[0]) if len(words) == 1 else Phrase(words)
        if kind == "LPAR":
            self.advance()
            node = self.or_expr()
            if self.peek() != "RPAR":
                raise QueryError("не закрито дужку")
            self.advance()
            return node
        raise QueryError(f"несподіваний елемент у запиті: {kind}")


def parse(query: str) -> Node:
    return _Parser(_lex(query)).parse()