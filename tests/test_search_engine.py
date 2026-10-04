import pytest

from researchxtract import config
from researchxtract.core import QuerySyntaxError, SearchEngine
from researchxtract.core.boolean_query import And, Not, Or, Term, parse_query
from researchxtract.core.corpus import iter_jsonl


@pytest.fixture(scope="module")
def engine():
    return SearchEngine.build(iter_jsonl(config.SAMPLE_CORPUS_PATH))


def ids(response):
    return {h.paper_id for h in response.hits}


def test_and_is_intersection(engine):
    assert ids(engine.search("CNN AND segmentation")) == {"toy-001", "toy-005"}


def test_or_is_union(engine):
    assert ids(engine.search("CNN OR transformer")) == {
        "toy-001", "toy-002", "toy-004", "toy-005", "toy-009"}


def test_not_is_difference(engine):
    cnn = ids(engine.search("CNN", 50))
    classification = ids(engine.search("classification", 50))
    assert ids(engine.search("CNN AND NOT classification", 50)) == cnn - classification
    assert "toy-004" not in ids(engine.search("CNN AND NOT classification"))


def test_implicit_not_means_and_not(engine):
    assert ids(engine.search("CNN NOT classification")) == ids(engine.search("CNN AND NOT classification"))


def test_parentheses_and_precedence():
    tree = parse_query("cnn OR transformer AND medical")
    assert isinstance(tree, Or) and isinstance(tree.children[1], And)
    tree = parse_query("(cnn OR transformer) AND medical")
    assert isinstance(tree, And) and isinstance(tree.children[0], Or)
    assert isinstance(parse_query("cnn AND NOT medical").children[1], Not)


def test_multi_concept_query_ranks_best_match_first(engine):
    assert engine.search("deep learning medical image segmentation").hits[0].paper_id == "toy-001"
    assert engine.search("UAV navigation GPS denied sensor fusion").hits[0].paper_id == "toy-003"
    assert engine.search("transformer time series anomaly detection").hits[0].paper_id == "toy-002"


def test_scores_sorted_descending_and_top_k_respected(engine):
    resp = engine.search("deep learning medical image segmentation", 3)
    scores = [h.score for h in resp.hits]
    assert len(resp.hits) == 3 and scores == sorted(scores, reverse=True)
    assert resp.total_matches >= 3


def test_unknown_stopword_and_empty_queries(engine):
    assert engine.search("zzzunknownterm").hits == []
    assert engine.search("the").hits == []
    assert engine.search("").hits == []
    assert engine.search("CNN AND zzzunknownterm").hits == []


def test_default_operator_and_is_stricter(engine):
    strict = SearchEngine(engine.index, default_operator="AND")
    assert ids(strict.search("cnn segmentation")) == {"toy-001", "toy-005"}
    assert len(engine.search("cnn segmentation", 50).hits) > 2


def test_malformed_queries_raise(engine):
    for bad in ("CNN AND", "(CNN OR transformer", "AND CNN", "CNN )"):
        with pytest.raises(QuerySyntaxError):
            engine.search(bad)


def test_idf_rare_terms_weigh_more(engine):
    idx = engine.index
    assert idx.doc_freq("kalman") < idx.doc_freq("medical")
