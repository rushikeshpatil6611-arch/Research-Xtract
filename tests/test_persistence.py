import os

from researchxtract import config
from researchxtract.core import SearchEngine
from researchxtract.core.corpus import iter_jsonl


def test_save_load_roundtrip_gives_identical_results(tmp_path):
    engine = SearchEngine.build(iter_jsonl(config.SAMPLE_CORPUS_PATH))
    path = tmp_path / "idx" / "index.json"
    engine.save(path)
    loaded = SearchEngine.load(path)
    for q in ("CNN AND segmentation", "deep learning medical image segmentation", "CNN OR transformer"):
        a, b = engine.search(q, 10), loaded.search(q, 10)
        assert [(h.paper_id, h.score) for h in a.hits] == [(h.paper_id, h.score) for h in b.hits]


def test_atomic_save_leaves_no_tmp_file(tmp_path):
    engine = SearchEngine.build(iter_jsonl(config.SAMPLE_CORPUS_PATH))
    path = tmp_path / "index.json"
    engine.save(path)
    assert path.exists() and not os.path.exists(str(path) + ".tmp")
