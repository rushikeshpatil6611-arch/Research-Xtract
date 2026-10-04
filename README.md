# ResearchXtract
Multi-Process Academic Information Retrieval Engine (Theme 2: Multi-Process Search Engine
with Persistent Index). Query-based retrieval over a local research-paper corpus -
**not** a recommendation system. OpenAlex is only the data source.

**Current status:** Phase 1 (DSA search core) implemented and unit-tested on a small
synthetic corpus. Phases 2-5 not started. No real-corpus benchmarks exist yet.

## Setup (VS Code, Windows or Linux)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/Mac:  source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
pytest                       # all tests should pass
```
In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> choose `.venv`.

## Run Phase 1
```bash
python scripts/build_index.py --corpus data/sample/toy_corpus.jsonl --index data/index/toy_index.json
python scripts/search_cli.py  --corpus data/sample/toy_corpus.jsonl --index data/index/toy_index.json
python scripts/search_cli.py  --corpus data/sample/toy_corpus.jsonl --index data/index/toy_index.json -q "CNN AND NOT classification"
```

## Folder map (who owns what)
```
researchxtract/
├── pyproject.toml, requirements.txt, .gitignore, README.md
├── src/researchxtract/
│   ├── config.py              # all tunable settings
│   ├── core/                  # PHASE 1 - A1 - DSA search engine core   [DONE]
│   │   ├── preprocessing.py   #   tokenize, stopwords, stemming (analyze)
│   │   ├── posting.py         #   PostingList + intersect/union/difference
│   │   ├── inverted_index.py  #   term -> PostingList hash map, doc stats
│   │   ├── tfidf.py           #   TF, IDF, doc norms, scoring, Top-K heap
│   │   ├── boolean_query.py   #   parser (AND/OR/NOT/parens) + evaluator
│   │   ├── search_engine.py   #   SearchEngine facade: build/search/save/load
│   │   ├── persistence.py     #   atomic JSON save/load (A3 hardens later)
│   │   └── corpus.py          #   JSONL corpus reader (contract with A2)
│   ├── data/                  # PHASE 2 - A2 - OpenAlex collector, cleaning, MySQL
│   ├── processes/             # PHASE 3 - A3 - Indexer, Query Handler, IPC, recovery
│   ├── api/                   # PHASE 4 - A4 - HTTP/JSON endpoint
│   └── evaluation/            # PHASE 5 - LIKE benchmark, precision/recall, latency
├── web/                       # PHASE 4 - HTML/CSS/JS search page
├── scripts/                   # build_index.py, search_cli.py (+ later run scripts)
├── tests/                     # pytest unit tests
├── data/
│   ├── raw/                   # raw OpenAlex responses (git-ignored)
│   ├── corpus/                # papers.jsonl - cleaned local corpus
│   ├── index/                 # persisted index files (git-ignored)
│   ├── eval/                  # 15+ labelled queries (relevance judgements)
│   └── sample/                # SYNTHETIC toy corpus for tests/demo only
└── docs/                      # design decisions, UML, report material
```

## Corpus contract (A2 -> A1)
One JSON object per line in `data/corpus/papers.jsonl`:
`{"paper_id": "...", "title": "...", "abstract": "...", "keywords": ["..."], "year": 2021}`

## Query syntax
`AND`, `OR`, `NOT` (UPPERCASE), parentheses. Precedence NOT > AND > OR.
No operator between words -> `config.DEFAULT_OPERATOR` (OR, ranked by TF-IDF).
