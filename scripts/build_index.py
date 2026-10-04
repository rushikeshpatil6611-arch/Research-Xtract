"""Build and save the index from a JSONL corpus.

    python scripts/build_index.py                          # uses data/corpus/papers.jsonl
    python scripts/build_index.py --corpus data/sample/toy_corpus.jsonl
"""
import argparse
import time

from researchxtract import config
from researchxtract.core import SearchEngine
from researchxtract.core.corpus import iter_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ResearchXtract inverted index")
    ap.add_argument("--corpus", default=str(config.CORPUS_PATH))
    ap.add_argument("--index", default=str(config.INDEX_PATH))
    args = ap.parse_args()

    t0 = time.perf_counter()
    engine = SearchEngine.build(iter_jsonl(args.corpus))
    t1 = time.perf_counter()
    engine.save(args.index)
    t2 = time.perf_counter()

    print(f"Indexed in {t1 - t0:.2f}s, saved in {t2 - t1:.2f}s -> {args.index}")
    for key, value in engine.index.stats().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
