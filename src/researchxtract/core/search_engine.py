"""SearchEngine: the single entry point other components use.

    engine = SearchEngine.build(documents)           # Indexer side
    engine.save(path)  /  SearchEngine.load(path)    # persistence
    response = engine.search("CNN AND segmentation") # Query Handler side

Query pipeline:
    query text -> parse (Boolean tree) -> evaluate on posting lists (candidate docs)
               -> TF-IDF score candidates -> Top-K -> SearchResponse
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .. import config
from .boolean_query import evaluate, parse_query, positive_terms
from .inverted_index import InvertedIndex
from .persistence import load_index, save_index
from .preprocessing import analyze
from .tfidf import score_candidates, top_k


@dataclass
class SearchHit:
    doc_id: int          # internal integer id
    paper_id: str        # external id (maps to MySQL paper_id)
    score: float


@dataclass
class SearchResponse:
    query: str
    total_matches: int
    hits: list[SearchHit] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:     # ready for the HTTP/JSON layer (A4)
        return asdict(self)


def document_to_terms(doc: dict) -> list[str]:
    """Searchable text = title + keywords + abstract. Boosts repeat a field's terms."""
    keywords = doc.get("keywords") or ""
    if isinstance(keywords, (list, tuple)):
        keywords = " ".join(str(k) for k in keywords)
    title_terms = analyze(doc.get("title") or "", config.USE_STEMMING)
    keyword_terms = analyze(keywords, config.USE_STEMMING)
    abstract_terms = analyze(doc.get("abstract") or "", config.USE_STEMMING)
    return title_terms * config.TITLE_BOOST + keyword_terms * config.KEYWORD_BOOST + abstract_terms


class SearchEngine:
    def __init__(self, index: InvertedIndex, default_operator: str = config.DEFAULT_OPERATOR):
        if not index.is_finalized:
            raise RuntimeError("Index must be finalized before searching")
        self.index = index
        self.default_operator = default_operator

    # ---- build / persist -------------------------------------------------
    @classmethod
    def build(cls, documents: Iterable[dict],
              default_operator: str = config.DEFAULT_OPERATOR) -> "SearchEngine":
        index = InvertedIndex()
        for doc in documents:
            index.add_document(str(doc["paper_id"]), document_to_terms(doc))
        index.finalize()
        return cls(index, default_operator)

    def save(self, path: str | Path = config.INDEX_PATH) -> None:
        save_index(self.index, path)

    @classmethod
    def load(cls, path: str | Path = config.INDEX_PATH,
             default_operator: str = config.DEFAULT_OPERATOR) -> "SearchEngine":
        return cls(load_index(path), default_operator)

    # ---- search ------------------------------------------------------------
    def search(self, query: str, top_k_results: int = config.DEFAULT_TOP_K) -> SearchResponse:
        start = time.perf_counter()
        tree = parse_query(query, self.default_operator)
        candidates = evaluate(tree, self.index) or []                  # Boolean retrieval
        scores = score_candidates(self.index, positive_terms(tree), candidates)  # TF-IDF
        ranked = top_k(scores, top_k_results)                          # Top-K
        hits = [SearchHit(d, self.index.external_ids[d], round(s, 6)) for d, s in ranked]
        elapsed = (time.perf_counter() - start) * 1000.0
        return SearchResponse(query, len(candidates), hits, round(elapsed, 3))
