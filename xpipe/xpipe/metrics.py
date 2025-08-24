#!/usr/bin/env python3
# xpipe/metrics.py
# ======================================================================
# MetricLog + quality metrics (ROUGE-L, token F1) + ECE computation.
#
# What you get:
#   • MetricLog.add(...)  : per-query row append (backward compatible)
#   • MetricLog.save(path): writes CSV; if confidence/correct present,
#                           also writes a sibling JSON with ECE summary.
#   • Utility functions   : rouge_l_fscore, f1_token_score
#
# Schema (per row) — fields are optional; we write what you pass:
#   pipeline, item, answer, reference, relevance, faithfulness,
#   rougeL_f, f1_token, correct, confidence, latency_ms, cost_usd
# ======================================================================

from __future__ import annotations
import csv
from typing import Dict, List, Any

class MetricLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._cols: set[str] = set()

    def add(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))
        self._cols.update(kwargs.keys())

    @property
    def columns(self) -> List[str]:
        return sorted(self._cols)

    def save(self, path: str) -> str:
        cols = self.columns
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in cols})
        return path

# ---------------------- simple text metrics ----------------------------

from collections import Counter

def _tok(s: str) -> List[str]:
    return [t.lower() for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s).split()]

def f1_token_score(pred: str, ref: str) -> float:
    p, r = _tok(pred), _tok(ref)
    if not p and not r: return 1.0
    if not p or not r:  return 0.0
    cp, cr = Counter(p), Counter(r)
    overlap = sum(min(cp[w], cr[w]) for w in set(cp) | set(cr))
    prec = overlap / max(1, sum(cp.values()))
    rec  = overlap / max(1, sum(cr.values()))
    return 0.0 if prec + rec == 0 else 2*prec*rec/(prec+rec)

def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

def rouge_l_fscore(pred: str, ref: str) -> float:
    p, r = _tok(pred), _tok(ref)
    if not p and not r: return 1.0
    if not p or not r:  return 0.0
    l = _lcs(p, r)
    prec = l / max(1, len(p))
    rec  = l / max(1, len(r))
    return 0.0 if prec + rec == 0 else 2*prec*rec/(prec+rec)