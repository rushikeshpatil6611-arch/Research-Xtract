"""Save / load the index to disk.

Phase 1 format: one JSON file (human-readable, easy to debug).
Writes are ATOMIC: data goes to "<file>.tmp", is flushed to disk (fsync), and only
then renamed over the real file with os.replace(). A crash mid-write therefore
leaves the OLD index intact instead of a half-written one.

A3 (Phase 3) will harden this (versioned index files, recovery on start-up,
crash tests) and may switch to a binary format for speed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .inverted_index import InvertedIndex


def save_index(index: InvertedIndex, path: str | Path) -> None:
    if not index.is_finalized:
        raise RuntimeError("Call index.finalize() before saving")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)              # atomic on Windows and Linux


def load_index(path: str | Path) -> InvertedIndex:
    with open(path, "r", encoding="utf-8") as f:
        return InvertedIndex.from_dict(json.load(f))
