"""TF-IDF scoring and Top-K selection.

Weight of term t in document d:
        w(t,d) = tf_weight(tf) * idf(t)

    tf_weight(tf) = 1 + log10(tf)         (sub-linear: 10 mentions != 10x more relevant)
    idf(t)        = log10(1 + N / df)     (rare term -> big idf; term in every doc -> small idf)

Score of document d for query q (cosine-style):
        score(q,d) = SUM over query terms t of  w(t,d) * idf(t)   /   ||d||

where ||d|| = sqrt(sum of w(t,d)^2 over all terms of d) is the document's vector
length. Dividing by it stops long documents from winning just by being long.

Information-theory view (Discrete Math link): idf(t) ~ log(1/P(t)) is the
"surprise"/information carried by seeing term t. Words that appear everywhere
carry almost no information, so they barely affect ranking.

PROPOSED: the "1 +" inside idf is a smoothing variant so a term present in every
document still gets a small positive weight instead of 0 (important on tiny corpora).
"""
from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .inverted_index import InvertedIndex
    from .posting import PostingList


def tf_weight(tf: int) -> float:
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


def idf(num_docs: int, df: int) -> float:
    return math.log10(1.0 + num_docs / df)


def compute_doc_norms(postings: "dict[str, PostingList]", num_docs: int) -> list[float]:
    """||d|| for every document. Runs once, after all documents are indexed."""
    squares = [0.0] * num_docs
    for plist in postings.values():
        term_idf = idf(num_docs, plist.df)
        for doc_id, tf in plist:
            w = tf_weight(tf) * term_idf
            squares[doc_id] += w * w
    return [math.sqrt(s) for s in squares]


def score_candidates(index: "InvertedIndex", query_terms: Iterable[str],
                     candidates: list[int]) -> dict[int, float]:
    """Score only the documents that Boolean retrieval already selected."""
    scores = dict.fromkeys(candidates, 0.0)
    if not scores:
        return scores
    for term in set(query_terms):                      # repeated query words count once
        plist = index.get_postings(term)
        if plist is None:
            continue
        term_idf = idf(index.num_docs, plist.df)
        for doc_id, tf in plist:
            if doc_id in scores:
                scores[doc_id] += tf_weight(tf) * term_idf * term_idf
    for doc_id in scores:
        norm = index.doc_norms[doc_id]
        if norm > 0.0:
            scores[doc_id] /= norm
    return scores


def top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Best k (doc_id, score) pairs using a heap: O(n log k), not a full O(n log n) sort.
    Ties are broken by smaller doc_id so results are deterministic."""
    return heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
