"""The inverted index - the heart of the search engine.

    term  ->  PostingList( doc_ids=[...sorted...], term_freqs=[...] )

`postings` is a Python dict, i.e. a HASH MAP: looking up a term is O(1) on
average, so a query never scans the whole corpus.

Internal doc ids are dense integers 0..N-1 assigned in insertion order.
`external_ids[doc_id]` maps back to the real paper id (OpenAlex id / MySQL
paper_id). Integers keep posting lists small and merge-able.

Lifecycle:  add_document(...) x N   ->   finalize()   ->   search / save
(Phase 1 index is immutable after finalize(). "Safe update" = build a new index
and atomically swap files - that is A3's Phase 3 job.)
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .posting import PostingList
from .tfidf import compute_doc_norms

FORMAT_VERSION = 1


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, PostingList] = {}   # the hash map: term -> posting list
        self.external_ids: list[str] = []            # doc_id -> paper id
        self.doc_lengths: list[int] = []             # doc_id -> number of indexed terms
        self.doc_norms: list[float] = []             # doc_id -> ||d|| (filled by finalize)
        self._finalized = False

    # ---- building -------------------------------------------------------
    def add_document(self, external_id: str, terms: Iterable[str]) -> int:
        """Index one document. `terms` is the output of preprocessing.analyze()."""
        if self._finalized:
            raise RuntimeError("Index is finalized; build a new index to add documents")
        terms = list(terms)
        doc_id = len(self.external_ids)
        self.external_ids.append(external_id)
        self.doc_lengths.append(len(terms))
        for term, tf in Counter(terms).items():       # TF = occurrences of term in this doc
            plist = self.postings.get(term)
            if plist is None:
                plist = self.postings[term] = PostingList()
            plist.add(doc_id, tf)                     # doc_id grows -> list stays sorted
        return doc_id

    def finalize(self) -> None:
        """Compute document norms (needs final N and every df). Call once after the last add."""
        self.doc_norms = compute_doc_norms(self.postings, self.num_docs)
        self._finalized = True

    # ---- reading --------------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.external_ids)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def get_postings(self, term: str) -> PostingList | None:
        return self.postings.get(term)

    def doc_freq(self, term: str) -> int:
        plist = self.postings.get(term)
        return plist.df if plist else 0

    def all_doc_ids(self) -> list[int]:
        return list(range(self.num_docs))

    def stats(self) -> dict:
        total_postings = sum(len(p) for p in self.postings.values())
        avg_len = sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        return {
            "documents": self.num_docs,
            "vocabulary_terms": len(self.postings),
            "total_postings": total_postings,
            "avg_doc_length_terms": round(avg_len, 2),
        }

    # ---- (de)serialisation, used by persistence.py -----------------------
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "external_ids": self.external_ids,
            "doc_lengths": self.doc_lengths,
            "doc_norms": self.doc_norms,
            "postings": {t: p.to_dict() for t, p in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InvertedIndex":
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported index format: {data.get('format_version')!r}")
        idx = cls()
        idx.external_ids = list(data["external_ids"])
        idx.doc_lengths = list(data["doc_lengths"])
        idx.doc_norms = list(data["doc_norms"])
        idx.postings = {t: PostingList.from_dict(p) for t, p in data["postings"].items()}
        idx._finalized = True
        return idx
