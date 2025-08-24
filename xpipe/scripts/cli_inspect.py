#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cli_inspect.py — Minimal, dependency-light inspector for X‑Pipe outputs.

Reads:
  - output/xpipe/metrics/*.csv
  - output/xpipe/runs/*.jsonl

Shows:
  - Per-run summary table (latency, cost, grounding / F1 / ROUGE if present)
  - Trade-off table (retriever × judge if columns exist)
  - Optional calibration curve + ECE if 'confidence' & 'correct' columns exist
  - Saves PNG plots under output/xpipe/figs/

Usage:
  python xpipe/scripts/cli_inspect.py
  python xpipe/scripts/cli_inspect.py --figs  # also write plots
"""
from __future__ import annotations
import os, glob, json, argparse, math
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "output/xpipe"
RUN_DIR = os.path.join(OUTDIR, "runs")
MET_DIR = os.path.join(OUTDIR, "metrics")
FIG_DIR = os.path.join(OUTDIR, "figs")

def read_metrics() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(MET_DIR, "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__run__"] = os.path.basename(f).replace(".csv","")
            dfs.append(df)
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def read_runs(limit: int = 3) -> List[List[Dict[str, Any]]]:
    files = sorted(glob.glob(os.path.join(RUN_DIR, "*.jsonl")))[-limit:]
    runs = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fin:
                evts = [json.loads(line) for line in fin if line.strip()]
            runs.append(evts)
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")
    return runs

def print_summary(df: pd.DataFrame):
    if df.empty:
        print("\n[info] no metrics CSVs found in output/xpipe/metrics/")
        return
    cols_maybe = ["relevance","rougeL_f","f1_token","latency_ms","cost_usd",
                  "synth_prompt_tokens","synth_completion_tokens",
                  "judge_prompt_tokens","judge_completion_tokens",
                  "retriever","judge_enabled","__run__"]
    for c in cols_maybe:
        if c not in df.columns:
            if c == "cost_usd":
                df[c] = 0.0  # older runs may not log cost
            elif c in ("retriever","judge_enabled"):
                df[c] = None

    agg = (df.groupby("__run__", dropna=False)
             .agg(mean_grounding=("relevance","mean"),
                  mean_rougeL=("rougeL_f","mean"),
                  mean_f1=("f1_token","mean"),
                  mean_latency_ms=("latency_ms","mean"),
                  total_cost_usd=("cost_usd","sum"),
                  synth_tokens=("synth_prompt_tokens","sum"),
                  judge_tokens=("judge_prompt_tokens","sum"))
             .reset_index())
    print("\n=== Per-run Summary ===")
    with pd.option_context('display.max_columns', None, 'display.width', 180):
        print(agg.to_string(index=False))

    # tradeoff slice (if columns exist)
    if {"retriever","judge_enabled"}.issubset(df.columns):
        slice_cols = ["retriever","judge_enabled","relevance","rougeL_f","f1_token","latency_ms","cost_usd","__run__"]
        trade = (df[slice_cols]
                 .groupby(["retriever","judge_enabled","__run__"], dropna=False)
                 .agg(mean_grounding=("relevance","mean"),
                      mean_rougeL=("rougeL_f","mean"),
                      mean_f1=("f1_token","mean"),
                      mean_latency_ms=("latency_ms","mean"),
                      total_cost_usd=("cost_usd","sum"))
                 .reset_index()
                 .sort_values(["retriever","judge_enabled","__run__"]))
        print("\n=== Retriever × Judge Trade-off ===")
        with pd.option_context('display.max_columns', None, 'display.width', 180):
            print(trade.to_string(index=False))

def plot_latency_vs_grounding(df: pd.DataFrame, outpath: str):
    ok = df.dropna(subset=["latency_ms","relevance"])
    if ok.empty:
        print("[plot] skip latency_vs_grounding (missing columns)")
        return
    plt.figure()
    plt.scatter(ok["latency_ms"], ok["relevance"], s=14)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Grounding / Relevance")
    plt.title("Latency vs Grounding")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[plot] wrote {outpath}")

def plot_mean_grounding_by_run(df: pd.DataFrame, outpath: str):
    if "__run__" not in df or "relevance" not in df:
        print("[plot] skip mean_grounding_by_run (missing columns)")
        return
    m = df.groupby("__run__")["relevance"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, max(2, 0.35*len(m))))
    m.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Grounding / Relevance")
    plt.title("Mean Grounding by Run")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[plot] wrote {outpath}")

def calibration_and_ece(df: pd.DataFrame, outpath_curve: str, outpath_table: str, bins: int = 10):
    """
    Computes calibration curve + ECE if the metrics table contains:
      - 'confidence' in [0,1]
      - either 'correct' in {0,1} or 'label' & 'pred' to deduce correctness

    If missing, we skip gracefully.
    """
    if "confidence" not in df.columns:
        print("[cal] no 'confidence' column — skipping calibration/ECE")
        return

    if "correct" in df.columns:
        use = df.dropna(subset=["confidence","correct"]).copy()
        use["correct"] = use["correct"].astype(float)
    elif {"label","pred"}.issubset(df.columns):
        use = df.dropna(subset=["confidence","label","pred"]).copy()
        use["correct"] = (use["label"] == use["pred"]).astype(float)
    else:
        print("[cal] need either 'correct' or ('label','pred') — skipping")
        return

    if use.empty:
        print("[cal] nothing to evaluate — skipping")
        return

    # bin by confidence
    use = use[(use["confidence"] >= 0.0) & (use["confidence"] <= 1.0)]
    use["bin"] = pd.cut(use["confidence"], bins=bins, labels=False, include_lowest=True)
    by = use.groupby("bin").agg(
        mean_conf=("confidence","mean"),
        acc=("correct","mean"),
        n=("correct","size")
    ).dropna()
    if by.empty:
        print("[cal] empty after binning — skipping")
        return

    # Expected Calibration Error
    total = by["n"].sum()
    by["abs_gap"] = (by["acc"] - by["mean_conf"]).abs()
    ece = (by["n"] * by["abs_gap"]).sum() / total

    # plot curve
    plt.figure()
    plt.plot(by["mean_conf"], by["acc"], marker="o")
    plt.plot([0,1],[0,1],"--", alpha=0.5)
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Calibration curve (ECE={ece:.3f})")
    os.makedirs(os.path.dirname(outpath_curve), exist_ok=True)
    plt.savefig(outpath_curve, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"[cal] wrote {outpath_curve}")

    # save table with bin stats + ECE row
    by2 = by.copy()
    by2.loc["ECE","mean_conf"] = float("nan")
    by2.loc["ECE","acc"] = float("nan")
    by2.loc["ECE","n"] = total
    by2.loc["ECE","abs_gap"] = ece
    by2.to_csv(outpath_table)
    print(f"[cal] wrote {outpath_table}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", action="store_true", help="also write PNG plots and calibration if possible")
    args = ap.parse_args()

    df = read_metrics()
    print_summary(df)

    if args.figs and not df.empty:
        plot_mean_grounding_by_run(df, os.path.join(FIG_DIR, "mean_grounding_by_run.png"))
        plot_latency_vs_grounding(df, os.path.join(FIG_DIR, "latency_vs_grounding.png"))
        calibration_and_ece(
            df,
            outpath_curve=os.path.join(FIG_DIR, "calibration_curve.png"),
            outpath_table=os.path.join(FIG_DIR, "calibration_bins.csv"),
            bins=10
        )

    # Optional: peek at last few run traces
    runs = read_runs(limit=2)
    for i, evts in enumerate(runs, 1):
        if not evts: continue
        print(f"\n=== Trace peek (run {i}, last 5 events) ===")
        for row in evts[-5:]:
            t = row.get("type","evt")
            nm = row.get("name","")
            dur = row.get("duration_ms", None)
            print(f"- {t:10s} {nm:20s} duration_ms={dur}")

if __name__ == "__main__":
    main()