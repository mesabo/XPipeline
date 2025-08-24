#!/usr/bin/env python3
# ======================================================================
# main.py — XPipe Runner (Release Stub)
# ----------------------------------------------------------------------
# This script is the entry point for XPipe experiments.
#
# Features:
#   • Loads experiment configs (YAML) + model registry (models.yaml).
#   • Runs a lightweight RAG (retrieval-augmented generation) pipeline.
#   • Supports Hugging Face (transformers) and optionally Ollama backends.
#   • Provides heuristic judging (unigram recall) or LLM-as-judge (optional).
#   • Traces every span for later inspection and saves metrics + logs.
#
# Outputs:
#   • JSONL run trace (plus pretty JSON copy for readability).
#   • CSV metrics file (per query).
#
# Usage:
#   python xpipe/main.py --config xpipe/configs/experiment_rag.yaml
#
# Notes:
#   • Free models only (e.g., distilgpt2, gpt2, Qwen-0.5B).
#   • Ollama section is optional (requires local daemon).
# ======================================================================

from __future__ import annotations
import argparse, os, time, yaml, json, datetime
from typing import Any, Dict, List, Tuple
from collections import Counter

# Core XPipe utilities
from xpipe.trace import Trace
from xpipe.metrics import MetricLog

# ----------------------------------------------------------------------
# Utility: Config Loading
# ----------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_configs(exp_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load experiment YAML and associated models.yaml.

    Search order:
      1) Explicit "models_path" inside experiment YAML.
      2) models.yaml in the same folder as exp_path.
      3) Canonical repo location: xpipe/configs/models.yaml
      4) Override via env: $XPIPE_MODELS

    Returns:
      (experiment_config, models_dict)
    """
    exp = load_yaml(exp_path)
    explicit = exp.get("models_path")
    if explicit and os.path.isfile(explicit):
        return exp, load_yaml(explicit)
    here = os.path.join(os.path.dirname(exp_path), "models.yaml")
    if os.path.isfile(here):
        return exp, load_yaml(here)
    repo_models = os.path.join("xpipe", "configs", "models.yaml")
    if os.path.isfile(repo_models):
        return exp, load_yaml(repo_models)
    env_models = os.getenv("XPIPE_MODELS")
    if env_models and os.path.isfile(env_models):
        return exp, load_yaml(env_models)
    raise FileNotFoundError("models.yaml not found. Checked explicit, alongside exp, repo default, and $XPIPE_MODELS.")

# ----------------------------------------------------------------------
# Simple Tokenization / Retrieval / Judging
# ----------------------------------------------------------------------
def _tokenize(s: str) -> List[str]:
    """Lowercased alphanumeric tokenizer (for heuristic retrieval/judging)."""
    return [t.lower() for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s).split()]

def retrieve_simple_overlap(query: str, corpus: List[Dict[str, str]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve top-k docs by lexical overlap score (unigram counts).
    """
    qtok = Counter(_tokenize(query))
    scored = []
    for doc in corpus:
        dtok = Counter(_tokenize(doc["text"]))
        overlap = sum(min(qtok[w], dtok[w]) for w in qtok)
        norm = max(1, sum(dtok.values()))
        score = overlap / norm
        scored.append({"id": doc["id"], "text": doc["text"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

def heuristic_grounding(answer: str, ctx_docs: List[str]) -> float:
    """
    Judge grounding via unigram recall:
      score = (# answer tokens appearing in context) / (# answer tokens)
    """
    a = _tokenize(answer)
    if not a: return 0.0
    ctx = Counter(_tokenize("\n".join(ctx_docs)))
    hit = sum(1 for w in a if ctx[w] > 0)
    return round(hit / len(a), 3)

# ----------------------------------------------------------------------
# LLM Backends (Hugging Face + optional Ollama)
# ----------------------------------------------------------------------
_HF_CACHE: Dict[str, Any] = {}

def _hf_generate(model_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a Hugging Face causal LM locally.
    Uses in-memory cache to avoid reloading model/tokenizer repeatedly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    mdl = _HF_CACHE.get(("mdl", model_id))
    tok = _HF_CACHE.get(("tok", model_id))
    if mdl is None or tok is None:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        mdl.eval()
        if torch.cuda.is_available():
            mdl.to("cuda")
        _HF_CACHE[("mdl", model_id)] = mdl
        _HF_CACHE[("tok", model_id)] = tok
    kwargs = dict(
        max_new_tokens=int(params.get("max_new_tokens", 160)),
        do_sample=bool(params.get("do_sample", False)),
        temperature=float(params.get("temperature", 0.0)),
        top_p=float(params.get("top_p", 1.0)),
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    ids = tok(prompt, return_tensors="pt")
    if mdl.device.type == "cuda":
        ids = {k: v.to("cuda") for k, v in ids.items()}
    t0 = time.time()
    with torch.no_grad():
        out = mdl.generate(**ids, **kwargs)
    dt = time.time() - t0
    full = tok.decode(out[0], skip_special_tokens=True)
    completion = full[len(prompt):] if full.startswith(prompt) else full
    return {
        "text": completion.strip(),
        "usage": {"prompt": len(tok.encode(prompt)), "completion": len(tok.encode(completion))},
        "latency_s": dt,
    }

def _ollama_generate(model_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run generation via Ollama (local server).
    Requires: `ollama serve` + `ollama pull <model>`.
    """
    import requests
    base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
    url = f"{base}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(params.get("temperature", 0.2)),
            "top_p": float(params.get("top_p", 0.9)),
            "num_predict": int(params.get("max_new_tokens", 200))
        }
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "text": data.get("response", "").strip(),
        "usage": {"prompt": data.get("prompt_eval_count", 0), "completion": data.get("eval_count", 0)},
        "latency_s": dt,
    }

def build_stage_runner(stage_cfg: Dict[str, Any], models: Dict[str, Any]):
    """
    Build a callable runner for a pipeline stage.

    stage_cfg:
      model: <handle in models.yaml>
      params: {overrides}

    Returns: function(prompt: str) -> dict
    """
    handle = stage_cfg["model"]
    if handle not in models:
        raise ValueError(f"Model handle '{handle}' not found in models.yaml.")
    entry = models[handle]
    backend = entry.get("backend", "hf")
    model_id = entry["id"]
    base = entry.get("params", {})
    overrides = stage_cfg.get("params", {})
    def _run(prompt: str, **kw):
        params = {**base, **overrides, **kw}
        if backend == "hf":
            return _hf_generate(model_id, prompt, params)
        elif backend == "ollama":
            return _ollama_generate(model_id, prompt, params)
        else:
            raise ValueError(f"Unsupported backend '{backend}' for handle '{handle}'")
    return _run

# ----------------------------------------------------------------------
# Pipeline: Retrieval-Augmented Generation (RAG)
# ----------------------------------------------------------------------
def run_rag(cfg: Dict[str, Any], models: Dict[str, Any], trace: Trace, metrics: MetricLog) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:
      1. Retrieve top-k docs from corpus (simple overlap).
      2. Synthesize answer with LLM.
      3. Judge grounding with heuristic unigram recall.
    """
    top_k = int(cfg["retriever"].get("top_k", 3))
    synth = build_stage_runner(cfg["llms"]["synthesize"], models)
    corpus = cfg["corpus"]
    results: Dict[str, Any] = {}
    for q in cfg["queries"]:
        qid, qtext = q["id"], q["text"]
        with trace.span("retrieve") as sp:
            top = retrieve_simple_overlap(qtext, corpus, k=top_k)
            sp.log({"k": len(top), "scores": [r["score"] for r in top]})
            ctx_docs = [r["text"] for r in top]
        with trace.span("synthesize") as sp:
            prompt = ("Answer the question using ONLY the evidence.\n"
                      f"QUESTION: {qtext}\n\nEVIDENCE:\n- " + "\n- ".join(ctx_docs) + "\n\nAnswer:")
            out = synth(prompt)
            trace.add_tokens(out["usage"]["prompt"], out["usage"]["completion"])
            sp.log({"latency_s": out.get("latency_s", None)})
            answer = out["text"]
        with trace.span("judge_heuristic") as sp:
            grounding = heuristic_grounding(answer, ctx_docs)
            sp.log({"grounding": grounding})
        metrics.add(pipeline="rag", item=qid, relevance=grounding,
                    faithfulness=grounding, latency_ms=trace.wall_time_ms, cost_usd=0.0)
        results[qid] = {"answer": answer, "grounding": grounding, "ctx_ids": [r["id"] for r in top]}
    return {"results": results}

PIPELINES = {"rag": run_rag}

# ----------------------------------------------------------------------
# CLI Entrypoint
# ----------------------------------------------------------------------
def auto_filename(prefix: str, suffix: str) -> str:
    """Generate timestamped filename in given directory prefix."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(prefix, f"{stamp}_{suffix}")

def main():
    """CLI wrapper for running configured XPipe pipelines."""
    ap = argparse.ArgumentParser(description="XPipe (free local) runner")
    ap.add_argument("--config", required=True, help="Experiment YAML (e.g. xpipe/configs/experiment_rag.yaml)")
    args = ap.parse_args()

    cfg, models = load_configs(args.config)

    # Ensure log dirs exist
    os.makedirs(cfg["logdir"], exist_ok=True)
    for sub in ("runs", "metrics", "ablations", "figs"):
        os.makedirs(os.path.join(cfg["logdir"], sub), exist_ok=True)

    trace = Trace(experiment=cfg["name"], run_tags={"pipeline": cfg["pipeline"]}, logdir=cfg["logdir"])
    metrics = MetricLog()

    if cfg["pipeline"] not in PIPELINES:
        raise SystemExit(f"Unknown pipeline '{cfg['pipeline']}'. Choose from {list(PIPELINES)}")

    result = PIPELINES[cfg["pipeline"]](cfg, models, trace, metrics)

    # Save run trace (both JSONL + pretty JSON)
    run_path = cfg["outputs"].get("run_jsonl") or auto_filename(os.path.join(cfg["logdir"], "runs"), f"{cfg['name']}.jsonl")
    trace_path = trace.save(run_path)
    try:
        pretty_path = os.path.splitext(trace_path)[0] + ".pretty.json"
        with open(trace_path, "r", encoding="utf-8") as fin, open(pretty_path, "w", encoding="utf-8") as fout:
            events = [json.loads(line) for line in fin if line.strip()]
            json.dump(events, fout, ensure_ascii=False, indent=2)
        print(f"[X-Pipe] pretty copy: {pretty_path}")
    except Exception as e:
        print(f"[X-Pipe] pretty copy failed: {e}")

    # Save metrics
    met_path = cfg["outputs"].get("metrics_csv") or os.path.join(cfg["logdir"], "metrics", f"{cfg['name']}.csv")
    metrics.save(met_path)

    print(f"[X-Pipe] saved run: {run_path}")
    print(f"[X-Pipe] saved metrics: {met_path}")
    print(f"[X-Pipe] queries: {list(result.get('results', {}).keys())}")

if __name__ == "__main__":
    main()