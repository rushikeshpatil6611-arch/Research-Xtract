"""Phase 1 - DSA / Search Core Engine (owner: A1).

Public API:
    SearchEngine.build(documents)  -> build an index in memory
    engine.search("cnn AND segmentation", top_k=10)
"""
from .search_engine import SearchEngine, SearchHit, SearchResponse
from .inverted_index import InvertedIndex
from .boolean_query import QuerySyntaxError

__all__ = ["SearchEngine", "SearchHit", "SearchResponse", "InvertedIndex", "QuerySyntaxError"]
