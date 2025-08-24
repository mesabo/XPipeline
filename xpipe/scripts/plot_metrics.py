#!/usr/bin/env python3
"""
plot_metrics.py
---------------------------------------------------------
Reads metrics CSVs produced by main.py and emits:
  - mean_grounding_by_run.png  (bar of mean 'relevance' per run)
  - latency_vs_grounding.png   (scatter of mean latency_ms vs mean relevance)
Usage:
  python xpipe/scripts/plot_metrics.py \
      --metrics_dir output/xpipe/metrics \
      --out_dir output/xpipe/figs
"""
from __future__ import annotations
import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(args.metrics_dir, "*.csv")))
    if not csvs:
        print(f"[plot] No CSVs in {args.metrics_dir}")
        return

    # Collect per-run aggregates
    rows = []
    for path in csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        # Expect columns from MetricLog.add(): pipeline,item,relevance,faithfulness,latency_ms,cost_usd
        if not {"relevance","latency_ms"}.issubset(df.columns):
            print(f"[plot] Skip {path}: missing required columns")
            continue
        rows.append({
            "run": name,
            "mean_relevance": df["relevance"].mean(),
            "mean_latency_ms": df["latency_ms"].mean(),
        })

    if not rows:
        print("[plot] No usable metrics found.")
        return

    agg = pd.DataFrame(rows).sort_values("mean_relevance", ascending=False)

    # Figure 1: mean grounding (= relevance) by run
    plt.figure(figsize=(10, 5))
    plt.bar(agg["run"], agg["mean_relevance"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Relevance (grounding proxy)")
    plt.title("Mean Grounding by Run")
    f1 = os.path.join(args.out_dir, "mean_grounding_by_run.png")
    plt.tight_layout()
    plt.savefig(f1, dpi=150)
    plt.close()

    # Figure 2: latency vs grounding
    plt.figure(figsize=(6, 5))
    plt.scatter(agg["mean_latency_ms"], agg["mean_relevance"])
    plt.xlabel("Mean Latency (ms)")
    plt.ylabel("Mean Relevance")
    plt.title("Latency vs Grounding")
    f2 = os.path.join(args.out_dir, "latency_vs_grounding.png")
    plt.tight_layout()
    plt.savefig(f2, dpi=150)
    plt.close()

    print(f"[plot] saved:\n - {f1}\n - {f2}")

if __name__ == "__main__":
    main()