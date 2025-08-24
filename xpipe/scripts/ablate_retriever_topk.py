#!/usr/bin/env python3
# ablate_retriever_topk.py
# ----------------------------------------------------------------------
# Simple ablation: vary retriever.top_k and record mean grounding.
# Reads a base experiment YAML, writes results to output/xpipe/ablations/topk.csv
# ----------------------------------------------------------------------
from __future__ import annotations
import os, sys, copy, json, subprocess, tempfile, csv, datetime
import yaml
from pathlib import Path

def load_yaml(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def save_yaml(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def run_once(python, main_py, cfg_path):
    cmd = [python, main_py, "--config", cfg_path]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr, file=sys.stderr)
        raise SystemExit(f"Run failed: {' '.join(cmd)}")
    # Try to infer metrics path from stdout; also parse from config safely
    return cp.stdout

def find_latest_metrics(metrics_dir: str, prefix: str):
    # Fall back to "most recent file starting with prefix"
    files = list(Path(metrics_dir).glob(f"{prefix}*.csv"))
    if not files:
        return None
    return str(sorted(files)[-1])

def parse_mean_grounding(csv_path: str):
    # Your MetricLog writes per-query rows; we average the "relevance" column
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "relevance" in df.columns:
        return float(df["relevance"].mean())
    # Fallback: try "faithfulness"
    if "faithfulness" in df.columns:
        return float(df["faithfulness"].mean())
    return float("nan")

def main():
    if len(sys.argv) < 2:
        print("Usage: python xpipe/scripts/ablate_retriever_topk.py configs/experiment_rag.yaml")
        sys.exit(1)

    base_cfg_path = sys.argv[1]
    cfg = load_yaml(base_cfg_path)
    logdir = cfg["logdir"]
    os.makedirs(os.path.join(logdir, "ablations"), exist_ok=True)

    python = sys.executable
    main_py = "xpipe/main.py"  # entrypoint
    topk_list = [1, 2, 3, 4, 5]

    rows = []
    for k in topk_list:
        cfg_k = copy.deepcopy(cfg)
        cfg_k["name"] = f"{cfg['name']}_ablate_topk{k}"
        cfg_k["retriever"]["top_k"] = int(k)
        # ensure outputs use auto file names
        cfg_k.setdefault("outputs", {})
        cfg_k["outputs"]["run_jsonl"] = ""
        cfg_k["outputs"]["metrics_csv"] = ""
        # write temp config
        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = os.path.join(td, f"{cfg_k['name']}.yaml")
            save_yaml(tmp_cfg, cfg_k)
            print(f"[ablation] running top_k={k}")
            out = run_once(python, main_py, tmp_cfg)

        # locate metrics CSV (latest with this prefix)
        metrics_dir = os.path.join(logdir, "metrics")
        mpath = find_latest_metrics(metrics_dir, cfg_k["name"])
        score = parse_mean_grounding(mpath) if mpath else float("nan")
        rows.append({"top_k": k, "mean_grounding": score, "metrics_csv": mpath or ""})
        print(f"[ablation] k={k} -> mean_grounding={score:.3f}  ({mpath})")

    # write grid CSV
    out_csv = os.path.join(logdir, "ablations", "topk.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["top_k", "mean_grounding", "metrics_csv"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ablation] wrote grid: {out_csv}")

if __name__ == "__main__":
    main()