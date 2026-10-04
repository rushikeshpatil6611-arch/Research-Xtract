"""Posting lists and the set operations Boolean retrieval is built on.

A posting list for term t = the documents containing t, plus how many times.
Doc ids are kept SORTED, which allows linear-time merge operations:

    AND -> intersect   (set intersection)
    OR  -> union       (set union)
    NOT -> difference  (set difference)

Each runs in O(m + n) with two pointers - no hashing, no nested loops.
"""
from __future__ import annotations

from typing import Iterator


class PostingList:
    """Sorted doc ids with a parallel list of term frequencies.

    doc_ids   = [2, 5, 9]
    term_freqs= [1, 3, 2]   # term occurs 1x in doc 2, 3x in doc 5, 2x in doc 9
    df (document frequency) = len(doc_ids) = 3
    """

    __slots__ = ("doc_ids", "term_freqs")

    def __init__(self, doc_ids: list[int] | None = None, term_freqs: list[int] | None = None):
        self.doc_ids: list[int] = doc_ids if doc_ids is not None else []
        self.term_freqs: list[int] = term_freqs if term_freqs is not None else []

    def add(self, doc_id: int, tf: int) -> None:
        """Append a posting. Doc ids must arrive in increasing order (keeps list sorted for free)."""
        if self.doc_ids and doc_id <= self.doc_ids[-1]:
            raise ValueError("doc_id must be strictly increasing within a posting list")
        self.doc_ids.append(doc_id)
        self.term_freqs.append(tf)

    @property
    def df(self) -> int:
        return len(self.doc_ids)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return zip(self.doc_ids, self.term_freqs)

    def to_dict(self) -> dict:
        return {"ids": self.doc_ids, "tfs": self.term_freqs}

    @classmethod
    def from_dict(cls, data: dict) -> "PostingList":
        return cls(list(data["ids"]), list(data["tfs"]))


# ---------------------------------------------------------------------------
# Set operations on SORTED lists of doc ids. Inputs are never modified.
# ---------------------------------------------------------------------------

def intersect(a: list[int], b: list[int]) -> list[int]:
    """AND: ids present in both lists."""
    i = j = 0
    na, nb = len(a), len(b)
    out: list[int] = []
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def union(a: list[int], b: list[int]) -> list[int]:
    """OR: ids present in either list (no duplicates)."""
    i = j = 0
    na, nb = len(a), len(b)
    out: list[int] = []
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: list[int], b: list[int]) -> list[int]:
    """NOT: ids in `a` that are NOT in `b`  (a \\ b)."""
    i = j = 0
    na, nb = len(a), len(b)
    out: list[int] = []
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x == y:
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


def intersect_many(lists: list[list[int]]) -> list[int]:
    """AND over many lists. Starts with the SHORTEST list so intermediate results stay small."""
    if not lists:
        return []
    ordered = sorted(lists, key=len)
    result = ordered[0]
    for other in ordered[1:]:
        if not result:
            break                      # early exit: nothing can match any more
        result = intersect(result, other)
    return result
