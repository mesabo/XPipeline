#!/usr/bin/env python3
# ======================================================================
# main.py — XPipe entrypoint (reads dataset.path; writes expected metrics)
# ======================================================================

from __future__ import annotations
import argparse, os, json, datetime, yaml
from typing import Any, Dict, List, Tuple

from xpipe.metrics import MetricLog
from xpipe.runners.rag import run as run_rag

def load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_configs(exp_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    exp = load_yaml(exp_path)
    # Resolve models.yaml (soft)
    here = os.path.join(os.path.dirname(exp_path), "models.yaml")
    models = load_yaml(here) if os.path.isfile(here) else {}
    return exp, models

def load_corpus(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    ds = cfg.get("dataset", {})
    if ds and ds.get("path"):
        path = ds["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"dataset.path not found: {path}")
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                o = json.loads(line)
                out.append({"id": str(o.get("id", len(out))), "text": str(o.get("text", ""))})
        if not out: raise ValueError(f"Empty dataset at {path}")
        return out
    # fallback: inline
    return cfg.get("corpus", [])

def auto_filename(prefix: str, suffix: str) -> str:
    os.makedirs(prefix, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(prefix, f"{stamp}_{suffix}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g., xpipe/configs/experiment_rag.yaml")
    args = ap.parse_args()

    cfg, models = load_configs(args.config)

    # Ensure dirs
    os.makedirs(cfg["logdir"], exist_ok=True)
    for sub in ("runs", "metrics", "ablations", "figs"):
        os.makedirs(os.path.join(cfg["logdir"], sub), exist_ok=True)

    corpus = load_corpus(cfg)
    queries = cfg.get("queries", [])
    if not queries: raise SystemExit("No queries provided in config.")

    metrics = MetricLog()

    if cfg["pipeline"] == "rag":
        result = run_rag(cfg, models, metrics, corpus, queries)
    else:
        raise SystemExit(f"Unknown pipeline '{cfg['pipeline']}'")

    # Save a minimal trace (per query)
    run_path = cfg["outputs"].get("run_jsonl") or auto_filename(os.path.join(cfg["logdir"], "runs"), f"{cfg['name']}.jsonl")
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, rec in result.get("results", {}).items():
            f.write(json.dumps({"qid": qid, **rec}, ensure_ascii=False) + "\n")

    # Save metrics
    met_path = cfg["outputs"].get("metrics_csv") or os.path.join(cfg["logdir"], "metrics", f"{cfg['name']}.csv")
    metrics.save(met_path)

    print(f"[X-Pipe] saved run: {run_path}")
    print(f"[X-Pipe] saved metrics: {met_path}")
    print(f"[X-Pipe] queries: {list(result.get('results', {}).keys())}")

if __name__ == "__main__":
    main()