# xpipe/data.py
# ----------------------------------------------------------------------
# Minimal dataset utilities for XPipe.
# - JSONL corpus loader expecting {"id": str, "text": str} per line.
# ----------------------------------------------------------------------
from __future__ import annotations
from typing import List, Dict, Any
import json

def load_jsonl_corpus(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL corpus in the form:
      {"id": "doc_id", "text": "document text ..."}
    Returns a list of dicts with 'id' and 'text'.
    """
    corpus: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" in obj and "text" in obj and isinstance(obj["text"], str):
                corpus.append({"id": str(obj["id"]), "text": obj["text"]})
    return corpus