# Phase 1 design decisions

Status labels follow the project knowledge document: CONFIRMED = fixed by the project
scope; PROPOSED = our choice, open to change; OPEN = still undecided.

| Decision | Status |
|---|---|
| Custom inverted index (hash map + posting lists), no Elasticsearch/Lucene/Whoosh | CONFIRMED |
| Boolean AND / OR / NOT via set operations | CONFIRMED |
| TF-IDF ranking, Top-K output | CONFIRMED |
| Searchable fields: title + abstract + keywords | CONFIRMED |
| Python as implementation language | PLANNED / RECOMMENDED |
| Hash map = Python `dict`; posting list = two parallel sorted lists | PROPOSED |
| Tokenizer: lowercase, `[a-z0-9]+`, hyphens split | PROPOSED |
| Light custom stemmer (not Porter) | PROPOSED |
| Stopword list (generic English) | PROPOSED - extend after seeing real abstracts |
| TF weight `1 + log10(tf)` | PROPOSED |
| IDF `log10(1 + N/df)` (smoothed) | PROPOSED |
| Cosine-style length normalisation (`/ ||d||`) | PROPOSED |
| Title and keyword boost x2 | PROPOSED - tune with precision/recall |
| Query with no operator = OR, ranked | PROPOSED - switch to AND in `config.py` if precision matters more |
| Operators must be UPPERCASE | PROPOSED |
| JSON index file with atomic write | PROPOSED placeholder - A3 finalises persistence |
| Index immutable after `finalize()` | PROPOSED - updates are a Phase 3 (A3) concern |
| Exact IPC mechanism | OPEN (Phase 3) |
| Web framework | OPEN (Phase 4) |
| Final relational schema | OPEN (Phase 2) |

## Not measured yet
No benchmark, precision/recall or latency numbers exist. The toy corpus in
`data/sample/` is SYNTHETIC test data written for unit tests; it is not real papers and
must never be used for reported results.
