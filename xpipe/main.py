#!/usr/bin/env python3
# ======================================================================
# main.py — XPipe Runner (Refactor: datasets/ + runners/)
# ----------------------------------------------------------------------
# What changed (compared to your current main.py):
#   1) No more inline RAG pipeline here. We now dispatch to xpipe.runners.*
#   2) Corpus is NOT embedded in config anymore. We load it from datasets/.
#   3) Config points to dataset JSONL via: cfg["dataset"]["path"].
#   4) Clear errors if dataset missing or empty.
#
# Usage:
#   python main.py --config configs/experiment_rag.yaml
#
# Outputs:
#   • JSONL run trace (+ pretty JSON)
#   • CSV metrics
# ======================================================================

from __future__ import annotations
import argparse, os, json, datetime, yaml
from typing import Any, Dict, Tuple, Callable

# Core XPipe utilities
from xpipe.trace import Trace
from xpipe.metrics import MetricLog
from xpipe.data import load_jsonl_corpus

# Map pipeline name -> runner callable
from xpipe.runners.rag import run as run_rag

PIPELINES: Dict[str, Callable[[Dict[str, Any], Dict[str, Any], Trace, MetricLog], Dict[str, Any]]] = {
    "rag": run_rag
}

# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_configs(exp_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (experiment_config, models_dict)

    Search order for models.yaml:
      1) Explicit exp["models_path"]
      2) <same-dir-as-exp>/models.yaml
      3) configs/models.yaml
      4) $XPIPE_MODELS
    """
    exp = load_yaml(exp_path)

    choices = []
    if exp.get("models_path"): choices.append(exp["models_path"])
    choices.append(os.path.join(os.path.dirname(exp_path), "models.yaml"))
    choices.append(os.path.join("configs", "models.yaml"))
    if os.getenv("XPIPE_MODELS"):
        choices.append(os.getenv("XPIPE_MODELS"))

    for candidate in choices:
        if candidate and os.path.isfile(candidate):
            return exp, load_yaml(candidate)

    raise FileNotFoundError("models.yaml not found via explicit path, alongside exp, configs/, or $XPIPE_MODELS")

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
def auto_filename(prefix: str, suffix: str) -> str:
    os.makedirs(prefix, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(prefix, f"{stamp}_{suffix}")

# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="XPipe runner")
    ap.add_argument("--config", required=True, help="Experiment YAML (e.g. configs/experiment_rag.yaml)")
    args = ap.parse_args()

    cfg, models = load_configs(args.config)

    # Create log dirs
    os.makedirs(cfg["logdir"], exist_ok=True)
    for sub in ("runs", "metrics", "ablations", "figs"):
        os.makedirs(os.path.join(cfg["logdir"], sub), exist_ok=True)

    # ---- Load dataset from datasets/ (required) ----
    if "dataset" not in cfg or "path" not in cfg["dataset"]:
        raise SystemExit("Config must include dataset.path pointing to a JSONL corpus under datasets/")

    ds_path = cfg["dataset"]["path"]
    if not os.path.isfile(ds_path):
        raise SystemExit(f"Dataset file not found: {ds_path}")

    corpus = load_jsonl_corpus(ds_path)  # List[{'id': str, 'text': str}]
    if not corpus:
        raise SystemExit(f"Dataset is empty: {ds_path}")

    # Queries are still from YAML (or you can also move them to datasets/)
    queries = cfg.get("queries", [])
    if not queries:
        raise SystemExit("No queries provided in config under 'queries:'")

    # Trace + metrics
    trace = Trace(experiment=cfg["name"], run_tags={"pipeline": cfg["pipeline"]}, logdir=cfg["logdir"])
    metrics = MetricLog()

    # Dispatch
    if cfg["pipeline"] not in PIPELINES:
        raise SystemExit(f"Unknown pipeline '{cfg['pipeline']}'. Choose from {list(PIPELINES)}")

    result = PIPELINES[cfg["pipeline"]](cfg, models, trace, metrics, corpus, queries)

    # Save run trace
    run_path = cfg.get("outputs", {}).get("run_jsonl") or auto_filename(os.path.join(cfg["logdir"], "runs"), f"{cfg['name']}.jsonl")
    trace_path = trace.save(run_path)
    try:
        pretty_path = os.path.splitext(trace_path)[0] + ".pretty.json"
        with open(trace_path, "r", encoding="utf-8") as fin, open(pretty_path, "w", encoding="utf-8") as fout:
            events = [json.loads(line) for line in fin if line.strip()]
            json.dump(events, fout, ensure_ascii=False, indent=2)
        print(f"[XPipe] pretty copy: {pretty_path}")
    except Exception as e:
        print(f"[XPipe] pretty copy failed: {e}")

    # Save metrics
    met_path = cfg.get("outputs", {}).get("metrics_csv") or os.path.join(cfg["logdir"], "metrics", f"{cfg['name']}.csv")
    metrics.save(met_path)

    print(f"[XPipe] saved run: {run_path}")
    print(f"[XPipe] saved metrics: {met_path}")
    print(f"[XPipe] queries: {list(result.get('results', {}).keys())}")

if __name__ == "__main__":
    main()