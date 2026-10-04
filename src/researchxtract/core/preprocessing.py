"""Text analysis: raw text -> list of index terms.

IMPORTANT RULE: documents and queries MUST go through exactly the same
pipeline. If the index stores "image" but the query looks up "images",
the search silently fails. Both sides call `analyze()`.

Pipeline:  lowercase -> tokenize -> drop stopwords/short tokens -> light stemming
"""
from __future__ import annotations

import re

# Letters+digits runs. "GPS-denied" -> ["gps", "denied"]; "3D" -> ["3d"].
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common words that carry no search value. (PROPOSED list - extend after
# looking at real OpenAlex abstracts, e.g. "paper", "propose", "result".)
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being between both but by could did do does doing during each few for
    from further had has have having he her here hers him his how i if in into
    is it its just me more most my no nor not of off on once only or other our
    out over own same she should so some such than that the their them then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def stem(token: str) -> str:
    """Very light suffix stripping so 'images' == 'image', 'learning' == 'learn'.

    This is NOT the full Porter stemmer; it is a small, predictable,
    explainable rule set. Because the SAME function is applied to documents and
    queries, odd stems (e.g. 'series' -> 'sery') are harmless.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"          # anomalies -> anomaly
    if token.endswith("sses"):
        return token[:-2]                # classes -> class
    if token.endswith(("ss", "us", "is")):
        return token                     # process, bus, analysis stay unchanged
    if token.endswith("s"):
        token = token[:-1]               # images -> image
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]  # learning -> learn, trained -> train
    return token


def analyze(text: str, use_stemming: bool = True) -> list[str]:
    """Full pipeline. Returns the list of terms (duplicates kept, so TF can be counted)."""
    terms: list[str] = []
    for tok in tokenize(text):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        terms.append(stem(tok) if use_stemming else tok)
    return terms
