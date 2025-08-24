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
import csv, json, os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------- tokenization ----------------------------
def _normalize_text(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in (s or ""))

def _tokens(s: str) -> List[str]:
    return [t for t in _normalize_text(s).split() if t]


# ---------------------------- ROUGE-L (LCS) ---------------------------
def _lcs(a: List[str], b: List[str]) -> int:
    # O(n*m) DP; fine for short answers
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        ai = a[i]
        for j in range(m):
            if ai == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l_fscore(pred: str, ref: str) -> float:
    """
    ROUGE-L F1 over token sequences (unigram LCS-based).
    Returns F1 in [0,1].
    """
    p_tok, r_tok = _tokens(pred), _tokens(ref)
    if not p_tok or not r_tok:
        return 0.0
    lcs = _lcs(p_tok, r_tok)
    prec = lcs / max(1, len(p_tok))
    rec  = lcs / max(1, len(r_tok))
    if prec + rec == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return float(round(f1, 6))


# ---------------------------- token F1 --------------------------------
def f1_token_score(pred: str, ref: str) -> float:
    """
    Token-level F1 (set-based): precision/recall over unique tokens.
    Not perfect for long-form, but a cheap proxy.
    """
    p_set, r_set = set(_tokens(pred)), set(_tokens(ref))
    if not p_set or not r_set:
        return 0.0
    tp = len(p_set & r_set)
    prec = tp / len(p_set)
    rec  = tp / len(r_set)
    if prec + rec == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return float(round(f1, 6))


# ---------------------------- ECE -------------------------------------
def _ece_from_rows(rows: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """
    Expected Calibration Error (ECE) using 'confidence' in [0,1] and binary 'correct'.
    We bin predictions by confidence, compute |acc - conf|, and average weighted by bin mass.
    Returns a dict with ece and per-bin stats.
    """
    bins = [{"low": i/n_bins, "high": (i+1)/n_bins, "n": 0, "acc": 0.0, "conf": 0.0} for i in range(n_bins)]
    total = 0
    # Aggregate
    for r in rows:
        if "confidence" not in r or "correct" not in r:
            continue
        conf = r["confidence"]
        corr = 1.0 if r["correct"] else 0.0
        if conf is None:
            continue
        conf = max(0.0, min(1.0, float(conf)))
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b]["n"] += 1
        bins[b]["acc"] += corr
        bins[b]["conf"] += conf
        total += 1
    # Compute per-bin means
    ece = 0.0
    for b in bins:
        if b["n"] > 0:
            b["acc"]  = b["acc"]  / b["n"]
            b["conf"] = b["conf"] / b["n"]
            weight = b["n"] / max(1, total)
            ece += weight * abs(b["acc"] - b["conf"])
    return {"ece": round(ece, 6), "n": total, "bins": bins}


# ---------------------------- MetricLog --------------------------------
@dataclass
class MetricLog:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, **kw):
        """
        Append a per-item metric row. All keys are optional.
        Common keys:
          pipeline, item, answer, reference,
          relevance, faithfulness,
          rougeL_f, f1_token, correct, confidence,
          latency_ms, cost_usd
        """
        self.rows.append(dict(kw))

    # Back-compat alias used in some older code in this repo:
    def log_row(self, **kw):
        self.add(**kw)

    def save(self, path: str, also_write_ece_json: bool = True, ece_bins: int = 10):
        """
        Save rows to CSV; if 'confidence' and 'correct' present in rows,
        also write an ECE JSON summary next to the CSV.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Collect header
        header = set()
        for r in self.rows:
            header.update(r.keys())
        header = sorted(header)

        # Write CSV
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

        # Optional ECE JSON
        if also_write_ece_json:
            # only compute if we have both fields
            has_conf = any("confidence" in r for r in self.rows)
            has_corr = any("correct" in r for r in self.rows)
            if has_conf and has_corr:
                ece = _ece_from_rows(self.rows, n_bins=ece_bins)
                jpath = os.path.splitext(path)[0] + "_ece.json"
                with open(jpath, "w", encoding="utf-8") as jf:
                    json.dump(ece, jf, ensure_ascii=False, indent=2)