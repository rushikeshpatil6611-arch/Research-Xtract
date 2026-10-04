"""Interactive command-line search (Phase 1 test bench).

    python scripts/search_cli.py --corpus data/sample/toy_corpus.jsonl --index data/index/toy_index.json
    python scripts/search_cli.py -q "CNN AND NOT classification"
"""
import argparse

from researchxtract import config
from researchxtract.core import QuerySyntaxError, SearchEngine
from researchxtract.core.corpus import iter_jsonl


def show(engine: SearchEngine, titles: dict, query: str, k: int) -> None:
    try:
        resp = engine.search(query, k)
    except QuerySyntaxError as exc:
        print(f"  Query error: {exc}")
        return
    print(f"  {resp.total_matches} matches ({resp.elapsed_ms} ms), showing top {len(resp.hits)}")
    for rank, hit in enumerate(resp.hits, start=1):
        print(f"  {rank:>2}. [{hit.score:.4f}] {hit.paper_id}  {titles.get(hit.paper_id, '')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(config.CORPUS_PATH), help="used only to show titles")
    ap.add_argument("--index", default=str(config.INDEX_PATH))
    ap.add_argument("-q", "--query", help="run one query and exit")
    ap.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    args = ap.parse_args()

    engine = SearchEngine.load(args.index)
    titles = {str(d["paper_id"]): d.get("title", "") for d in iter_jsonl(args.corpus)}

    if args.query:
        show(engine, titles, args.query, args.k)
        return
    print("ResearchXtract search. Operators: AND OR NOT ( ).  Empty line to quit.")
    while True:
        q = input("search> ").strip()
        if not q:
            break
        show(engine, titles, q, args.k)


if __name__ == "__main__":
    main()
