"""Boolean query language: parse text -> tree -> posting-list set operations.

Supported syntax (operators must be UPPERCASE so ordinary words like "and" stay words):

    CNN AND segmentation            intersection
    CNN OR transformer              union
    CNN AND NOT classification      set difference
    (CNN OR transformer) AND medical
    deep learning medical image     no operators -> DEFAULT operator (config.DEFAULT_OPERATOR)

Precedence (high -> low):  NOT  >  AND  >  OR   (parentheses override).

Grammar:
    or_expr  := and_expr ( "OR" and_expr )*
    and_expr := not_expr ( "AND" not_expr )*
    not_expr := "NOT" not_expr | primary
    primary  := "(" or_expr ")" | WORD

Two steps: parse_query() builds an expression tree; evaluate() runs it on the index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .inverted_index import InvertedIndex
from .posting import difference, intersect_many, union
from .preprocessing import analyze

OPERATORS = {"AND", "OR", "NOT"}


class QuerySyntaxError(ValueError):
    """Raised for malformed queries (unbalanced parentheses, dangling operator...)."""


# ---- expression tree --------------------------------------------------------
@dataclass(frozen=True)
class Term:
    terms: tuple[str, ...]          # analysed terms; >1 only for hyphenated words (ANDed). () = stopword

@dataclass(frozen=True)
class And:
    children: tuple

@dataclass(frozen=True)
class Or:
    children: tuple

@dataclass(frozen=True)
class Not:
    child: object


# ---- lexing / parsing -------------------------------------------------------
def _lex(query: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", query)


def _ends_operand(tok: str) -> bool:
    return tok == ")" or (tok not in OPERATORS and tok != "(")


def _starts_operand(tok: str) -> bool:
    return tok == "(" or tok == "NOT" or (tok not in OPERATORS and tok != ")")


def _insert_implicit_operators(tokens: list[str], default_op: str) -> list[str]:
    """'a b' -> 'a <default_op> b'.   'a NOT b' -> 'a AND NOT b'."""
    out: list[str] = []
    for tok in tokens:
        if out and _ends_operand(out[-1]) and _starts_operand(tok):
            out.append("AND" if tok == "NOT" else default_op)
        out.append(tok)
    return out


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def parse(self):
        node = self._or()
        if self._peek() is not None:
            raise QuerySyntaxError(f"Unexpected token {self._peek()!r}")
        return node

    def _or(self):
        children = [self._and()]
        while self._peek() == "OR":
            self.pos += 1
            children.append(self._and())
        return children[0] if len(children) == 1 else Or(tuple(children))

    def _and(self):
        children = [self._not()]
        while self._peek() == "AND":
            self.pos += 1
            children.append(self._not())
        return children[0] if len(children) == 1 else And(tuple(children))

    def _not(self):
        if self._peek() == "NOT":
            self.pos += 1
            return Not(self._not())
        return self._primary()

    def _primary(self):
        tok = self._peek()
        if tok is None:
            raise QuerySyntaxError("Query ended unexpectedly (missing term after operator?)")
        if tok == "(":
            self.pos += 1
            node = self._or()
            if self._peek() != ")":
                raise QuerySyntaxError("Missing closing parenthesis")
            self.pos += 1
            return node
        if tok in OPERATORS or tok == ")":
            raise QuerySyntaxError(f"Unexpected {tok!r}")
        self.pos += 1
        return Term(tuple(analyze(tok)))        # same pipeline as the indexer


def parse_query(query: str, default_operator: str = "OR"):
    """Return an expression tree, or None for an empty query."""
    if default_operator not in ("AND", "OR"):
        raise ValueError("default_operator must be 'AND' or 'OR'")
    tokens = _lex(query)
    if not tokens:
        return None
    return _Parser(_insert_implicit_operators(tokens, default_operator)).parse()


# ---- evaluation ---------------------------------------------------------------
def evaluate(node, index: InvertedIndex) -> list[int] | None:
    """Return sorted matching doc ids.  None = 'no constraint' (e.g. the node was only a
    stopword); such nodes are ignored by AND / OR / NOT. Returned lists are read-only."""
    if node is None:
        return None

    if isinstance(node, Term):
        if not node.terms:
            return None
        lists = []
        for t in node.terms:
            plist = index.get_postings(t)
            if plist is None:
                return []                       # unknown term -> nothing can match
            lists.append(plist.doc_ids)
        return intersect_many(lists)

    if isinstance(node, Or):
        results = [r for r in (evaluate(c, index) for c in node.children) if r is not None]
        if not results:
            return None
        out = results[0]
        for r in results[1:]:
            out = union(out, r)
        return out

    if isinstance(node, And):
        positives = [c for c in node.children if not isinstance(c, Not)]
        negatives = [c.child for c in node.children if isinstance(c, Not)]
        pos_results = [r for r in (evaluate(c, index) for c in positives) if r is not None]
        if pos_results:
            result = intersect_many(pos_results)
        elif negatives:
            result = index.all_doc_ids()        # only NOT terms: start from "everything"
        else:
            return None
        for neg in negatives:                   # A AND NOT B  ==  A \ B
            neg_res = evaluate(neg, index)
            if neg_res is not None:
                result = difference(result, neg_res)
        return result

    if isinstance(node, Not):                   # bare NOT x  ==  universe \ x
        inner = evaluate(node.child, index)
        return None if inner is None else difference(index.all_doc_ids(), inner)

    raise TypeError(f"Unknown node type: {type(node)!r}")


def positive_terms(node, _negated: bool = False) -> list[str]:
    """Terms that should contribute to the TF-IDF score (terms under NOT are excluded)."""
    if node is None:
        return []
    if isinstance(node, Term):
        return [] if _negated else list(node.terms)
    if isinstance(node, Not):
        return positive_terms(node.child, not _negated)
    out: list[str] = []
    for child in node.children:
        out.extend(positive_terms(child, _negated))
    return out
