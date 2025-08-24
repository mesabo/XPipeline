#!/usr/bin/env python3
# scripts/build_tbmp_dataset.py
# ----------------------------------------------------------------------
# Build a tiny, reproducible public corpus from the TBMP (USPTO) PDF.
# - Downloads the June 2024 TBMP PDF (public).
# - Extracts text and writes a small JSONL corpus under datasets/tbmp_2024/.
# - Each JSONL line: {"id": "d<idx>", "text": "<paragraph or chunk>"}
#
# Notes:
#   • Keeps this intentionally small (few dozen chunks) for quick local runs.
#   • You can change CHUNK_COUNT / MAX_PAGES to trade speed vs recall.
# ----------------------------------------------------------------------
from __future__ import annotations
import os, json, re, textwrap
from io import BytesIO
from typing import List, Dict, Any

import requests
from pypdf import PdfReader

OUT_DIR = os.path.join("datasets", "tbmp_2024")
OUT_FILE = os.path.join(OUT_DIR, "chunks.jsonl")

TBMP_URL = "https://www.uspto.gov/sites/default/files/documents/tbmp-Master-June2024.pdf"

MAX_PAGES = 60          # keep small for demo; increase if you want more coverage
CHUNK_COUNT = 40        # number of chunks to write
MIN_CHARS_PER_CHUNK = 500

def fetch_pdf(url: str) -> PdfReader:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return PdfReader(BytesIO(r.content))

def extract_text(reader: PdfReader, max_pages: int) -> str:
    pages = min(max_pages, len(reader.pages))
    buf = []
    for p in range(pages):
        t = reader.pages[p].extract_text() or ""
        buf.append(t)
    return "\n".join(buf)

def normalize_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def split_into_chunks(text: str, n_chunks: int) -> List[str]:
    """
    Sentence-ish chunking by paragraphs; then re-pack to ~n_chunks.
    """
    paras = [normalize_spaces(p) for p in re.split(r"\n{2,}", text) if p.strip()]
    # concat small paras into larger ones until we have ~n_chunks
    total = " ".join(paras)
    approx = max(1, len(total) // n_chunks)
    # Greedy pack by characters
    chunks = []
    cur = []
    cur_len = 0
    for p in paras:
        if cur_len + len(p) < approx or cur_len < MIN_CHARS_PER_CHUNK:
            cur.append(p); cur_len += len(p)
        else:
            chunks.append(" ".join(cur)); cur, cur_len = [p], len(p)
    if cur:
        chunks.append(" ".join(cur))
    # Trim to n_chunks
    if len(chunks) > n_chunks:
        # merge tail
        merged = chunks[:n_chunks-1]
        merged.append(" ".join(chunks[n_chunks-1:]))
        chunks = merged
    return [c for c in chunks if c.strip()]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[build_tbmp_dataset] downloading {TBMP_URL}")
    reader = fetch_pdf(TBMP_URL)
    print(f"[build_tbmp_dataset] pages in PDF: {len(reader.pages)}; extracting up to {MAX_PAGES}")
    text = extract_text(reader, MAX_PAGES)
    if not text.strip():
        raise SystemExit("Failed to extract text from PDF.")

    chunks = split_into_chunks(text, CHUNK_COUNT)
    if not chunks:
        raise SystemExit("No chunks produced.")

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for i, c in enumerate(chunks):
            obj = {"id": f"d{i}", "text": c}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[build_tbmp_dataset] wrote {len(chunks)} chunks -> {OUT_FILE}")

if __name__ == "__main__":
    main()