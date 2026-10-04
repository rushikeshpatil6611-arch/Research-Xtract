"""Central settings. Every tunable value lives here so experiments are reproducible.

Values marked PROPOSED are design choices made by us (not fixed by the course
requirements). Tune them later using the precision/recall evaluation.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus" / "papers.jsonl"      # produced by A2 (Phase 2)
SAMPLE_CORPUS_PATH = DATA_DIR / "sample" / "toy_corpus.jsonl"  # synthetic, for tests/demo only
INDEX_PATH = DATA_DIR / "index" / "index.json"          # produced by the Indexer

# PROPOSED: how much extra weight a term gets when it appears in a more important field.
# Implemented by repeating the field's terms, so TF reflects it automatically.
TITLE_BOOST = 2
KEYWORD_BOOST = 2

# PROPOSED: what a query like "cnn segmentation" (no operators) means.
# "OR"  -> union of posting lists, ranked by TF-IDF (docs matching more terms score higher)
# "AND" -> intersection (stricter, higher precision, lower recall)
DEFAULT_OPERATOR = "OR"

DEFAULT_TOP_K = 10
USE_STEMMING = True
