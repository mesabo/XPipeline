#!/usr/bin/env python3
"""
ablate_retriever_topk.py
---------------------------------------------------------
Grid ablation over retriever.top_k. For each k:
  - Write a temp config derived from the base experiment YAML
  - Set retriever.top_k = k
  - Inject models_path so main.py can find the registry
  - Call xpipe/main.py with that temp config
Outputs a CSV "topk.csv" summarizing mean grounding per run.

Usage:
  python xpipe/scripts/ablate_retriever_topk.py \
      --exp xpipe/configs/experiment_rag.yaml \
      --models xpipe/configs/models.yaml \
      --main xpipe/main.py \
      --out output/xpipe/ablations/topk.csv \
      --ks 1 2 3 4 5
"""
from __future__ import annotations
import argparse, os, sys, yaml, tempfile, subprocess, glob, csv
import pandas as pd

def load_yaml(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, p: str):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def run_once(python_bin: str, main_py: str, cfg_path: str) -> int:
    return subprocess.call([python_bin, main_py, "--config", cfg_path])

def mean_grounding_from_metrics(metrics_dir: str, run_name_prefix: str) -> float | None:
    """
    Find metrics CSV that main.py wrote for this run name and compute mean 'relevance' column.
    """
    pattern = os.path.join(metrics_dir, f"{run_name_prefix}.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    # main.py MetricLog uses columns like: pipeline,item,relevance,faithfulness,latency_ms,cost_usd
    if "relevance" not in df.columns:
        return None
    return float(df["relevance"].mean())

def sanitize(s: str) -> str:
    import re
    s = s.lower().replace("/", "_").replace(":", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Base experiment YAML (e.g., xpipe/configs/experiment_rag.yaml)")
    ap.add_argument("--models", required=True, help="models.yaml path")
    ap.add_argument("--main", required=True, help="Path to xpipe/main.py (entrypoint)")
    ap.add_argument("--out", required=True, help="Output CSV for ablation summary")
    ap.add_argument("--ks", nargs="*", type=int, default=[1,2,3,4,5], help="List of k values")
    ap.add_argument("--python", default=sys.executable, help="Python binary")
    args = ap.parse_args()

    if not os.path.isfile(args.exp):    sys.exit(f"[ablation] ERROR: missing exp file: {args.exp}")
    if not os.path.isfile(args.models): sys.exit(f"[ablation] ERROR: missing models file: {args.models}")
    if not os.path.isfile(args.main):   sys.exit(f"[ablation] ERROR: missing main entrypoint: {args.main}")

    base = load_yaml(args.exp)

    # Ensure log dirs (mirrors main.py)
    logdir = base.get("logdir", "output/xpipe")
    runs_dir   = os.path.join(logdir, "runs")
    metrics_dir= os.path.join(logdir, "metrics")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for k in args.ks:
        print(f"[ablation] running top_k={k}")
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg.setdefault("retriever", {})["top_k"] = int(k)
        # Critical: tell main.py where the registry is
        cfg["models_path"] = os.path.abspath(args.models)

        # Make a unique run name
        base_name = cfg.get("name", "xpipe_rag")
        run_name = f"{base_name}_ablate_topk{k}"
        cfg["name"] = run_name

        # Write temp config and invoke xpipe/main.py
        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = os.path.join(td, f"{sanitize(run_name)}.yaml")
            dump_yaml(cfg, tmp_cfg)
            rc = run_once(args.python, args.main, tmp_cfg)
            if rc != 0:
                print(f"[ablation] WARN: run failed for k={k} (rc={rc})")
                rows.append({"top_k": k, "mean_grounding": None, "status": f"rc={rc}"})
                continue

        # After run, a metrics CSV should exist at metrics/<name>.csv
        mg = mean_grounding_from_metrics(metrics_dir, run_name)
        rows.append({"top_k": k, "mean_grounding": mg, "status": "ok" if mg is not None else "no-metrics"})

    # Write summary CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["top_k", "mean_grounding", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[ablation] summary → {args.out}")

if __name__ == "__main__":
    main()