"""Corpus reader. The corpus is a JSONL file: ONE JSON object per line.

Contract that A2's OpenAlex collector must produce (PROPOSED field names):
    {"paper_id": "W123...", "title": "...", "abstract": "...",
     "keywords": ["...", "..."], "year": 2021}
Only paper_id / title / abstract / keywords are read by the search core.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
