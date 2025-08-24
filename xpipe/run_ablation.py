# xpipe/run_ablation.py
# Ablation over RAG variants using your existing code (no new deps needed).
from __future__ import annotations
import os, copy, json, statistics as stats
import pandas as pd

from xpipe.attribution import ablate
from xpipe.trace import Trace
from xpipe.metrics import MetricLog
# We import PIPELINES + load_configs from main.py to reuse your RAG runner exactly as-is
from xpipe.main import load_configs, PIPELINES

# ---- CONFIG: point to a base experiment YAML you already use ----
BASE_EXP = "xpipe/configs/experiment_rag.yaml"  # change if yours lives elsewhere

# ---- WHAT TO SWEEP ----
# Use model *handles* from your models.yaml (hf/* and/or ollama/*)
SYNTH_HANDLES = [
    "hf/distilgpt2",
    "hf/gpt2",
    # Add more handles you’ve defined in models.yaml (all free/local)
    "hf/Qwen2.5-0.5B-Instruct",
    # "ollama/llama3.2-3b",   # enable if you run Ollama locally
    # "ollama/qwen2.5-3b",
]

FACTORS = {
    "synth_model": SYNTH_HANDLES,
    "top_k": [2, 3],          # try different retrieval fanout
    "temperature": [0.0, 0.2] # light decoding change
}

# ---- DATASET: which queries to run (by id from your experiment YAML) ----
# If your YAML already lists multiple queries, you can identify them here by ID
# or simply pass e.g. ["*"] to run all. For speed, use a subset first.
DATASET_QUERY_IDS = ["q1", "q2", "q3"]  # adjust to match your cfg["queries"] ids

# ---- OUTPUTS ----
LOGDIR = "output/xpipe"
ABLATION_DIR = os.path.join(LOGDIR, "ablations")
os.makedirs(ABLATION_DIR, exist_ok=True)
OUT_CSV = os.path.join(ABLATION_DIR, "rag_ablation.csv")

def _filter_queries(cfg, wanted_ids):
    if wanted_ids == ["*"]:
        return cfg
    keep = [q for q in cfg["queries"] if q["id"] in wanted_ids]
    cfg2 = copy.deepcopy(cfg)
    cfg2["queries"] = keep
    return cfg2

def _override_cfg(cfg, synth_model: str, top_k: int, temperature: float):
    cfg2 = copy.deepcopy(cfg)
    # retrieval fanout
    cfg2.setdefault("retriever", {})
    cfg2["retriever"]["top_k"] = int(top_k)

    # change synthesizer handle + temperature override
    cfg2.setdefault("llms", {}).setdefault("synthesize", {})
    cfg2["llms"]["synthesize"]["model"] = synth_model
    cfg2["llms"]["synthesize"].setdefault("params", {})
    cfg2["llms"]["synthesize"]["params"]["temperature"] = float(temperature)

    # ensure logdir exists
    cfg2["logdir"] = LOGDIR
    # keep runs pretty JSONs enabled (your main.py already handles it)
    return cfg2

def pipeline_fn(item_query_id: str, **variant):
    """
    item_query_id: one query id from DATASET_QUERY_IDS
    variant: dict with keys from FACTORS (synth_model, top_k, temperature)
    Returns a dict with summary stats (merged into ablation table).
    """
    base_cfg, models = load_configs(BASE_EXP)
    # shrink to one query so each row is “per (variant, query)”
    cfg = _filter_queries(base_cfg, [item_query_id])
    cfg = _override_cfg(cfg,
                        synth_model=variant["synth_model"],
                        top_k=variant["top_k"],
                        temperature=variant["temperature"])

    trace = Trace(experiment=cfg["name"], run_tags={"pipeline": cfg["pipeline"]}, logdir=cfg["logdir"])
    metrics = MetricLog()

    if cfg["pipeline"] not in PIPELINES:
        raise SystemExit(f"Unknown pipeline '{cfg['pipeline']}'. Choose from {list(PIPELINES)}")

    # Run the RAG pipeline (uses your main.py PIPELINES['rag'] which returns per‑query results)
    result = PIPELINES[cfg["pipeline"]](cfg, models, trace, metrics)

    # Aggregate: for this single query, pull grounding and token/latency
    # (Your RAG returns {"results": {qid: {"answer","grounding","ctx_ids"}}})
    qid = next(iter(result["results"].keys()))
    grounding = float(result["results"][qid]["grounding"])

    # Persist the run JSONL (+ pretty copy) & metrics CSV are handled in main.py normally,
    # but here we can still save metrics quickly if you want a per‑row CSV snapshot:
    metrics_path = os.path.join(LOGDIR, "metrics", f"ablation_{cfg['name']}_{qid}.csv")
    metrics.save(metrics_path)

    # Summaries for the ablation row
    return {
        "query_id": qid,
        "grounding": grounding,
        "wall_time_ms": trace.wall_time_ms,
        "prompt_tokens": trace.token_usage["prompt"],
        "completion_tokens": trace.token_usage["completion"],
    }

def merge_result(out: dict) -> dict:
    # ablate() calls this—just pass data through (already dict)
    return out

def main():
    # Dataset = list of query ids (strings)
    dataset = DATASET_QUERY_IDS[:]
    df = ablate(pipeline_fn, dataset, FACTORS, merge_result=merge_result)

    # Helpful derived columns
    if not df.empty:
        df["throughput_qps"] = 1000.0 / df["wall_time_ms"].clip(lower=1)
        # group summaries (optional): avg per synth model
        grp = (df.groupby(["synth_model"])
                 .agg(avg_grounding=("grounding", "mean"),
                      avg_wall_time_ms=("wall_time_ms", "mean"))
                 .reset_index())
        print("\n=== Per‑model summary ===")
        print(grp.to_string(index=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\n[ABLATION] wrote {OUT_CSV}")
    if not df.empty:
        print(f"[ABLATION] rows: {len(df)} | columns: {list(df.columns)}")

if __name__ == "__main__":
    main()