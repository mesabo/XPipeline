# xpipe/runners/rag.py
# ----------------------------------------------------------------------
# XPipe RAG Runner (robust prompt budgeting)
# - Retrieval: simple lexical overlap
# - Synthesis: HF or Ollama via models.yaml dispatch
# - Judge: heuristic unigram recall
# - NEW: prompt/token budgeting to avoid model context overflows
#
# I/O:
#   run(cfg, models, trace, metrics, corpus, queries) -> {"results": {...}}
# ----------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Any, List, Optional
from collections import Counter
import time

from xpipe.trace import Trace
from xpipe.metrics import MetricLog

# --- HF + optional Ollama backends ---
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, requests

_HF_CACHE = {}

# ------------------------ HF / Ollama backends ------------------------
def _hf_generate(model_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
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

    gen_kwargs = dict(
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
        out = mdl.generate(**ids, **gen_kwargs)
    latency = time.time() - t0
    full = tok.decode(out[0], skip_special_tokens=True)
    completion = full[len(prompt):] if full.startswith(prompt) else full
    return {
        "text": completion.strip(),
        "usage": {"prompt": len(tok.encode(prompt)), "completion": len(tok.encode(completion))},
        "latency_s": latency,
    }

def _ollama_generate(model_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = params.get("ollama_base") or "http://localhost:11434"
    url = f"{base}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(params.get("temperature", 0.2)),
            "top_p": float(params.get("top_p", 0.9)),
            "num_predict": int(params.get("max_new_tokens", 200)),
        },
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "text": data.get("response", "").strip(),
        "usage": {"prompt": data.get("prompt_eval_count", 0), "completion": data.get("eval_count", 0)},
        "latency_s": latency,
    }

def build_stage_runner(stage_cfg: Dict[str, Any], models: Dict[str, Any]):
    """
    stage_cfg:
      model: <handle in models.yaml>
      params: {overrides}
    models[handle]:
      backend: "hf" | "ollama"
      id: actual model id
      params: base defaults
    """
    handle = stage_cfg["model"]
    if handle not in models:
        raise ValueError(f"Model handle '{handle}' not in models.yaml")
    entry = models[handle]
    backend = entry.get("backend", "hf")
    model_id = entry["id"]
    base_params = entry.get("params", {})
    overrides = stage_cfg.get("params", {})

    def _run(prompt: str, **kw):
        params = {**base_params, **overrides, **kw}
        if backend == "hf":
            return _hf_generate(model_id, prompt, params)
        elif backend == "ollama":
            return _ollama_generate(model_id, prompt, params)
        else:
            raise ValueError(f"Unsupported backend '{backend}' for handle '{handle}'")
    return _run

# ------------------------ retrieval + judge ---------------------------
def _tokenize(s: str):
    return [t.lower() for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s).split()]

def retrieve_simple_overlap(query: str, corpus: List[Dict[str, str]], k: int = 3):
    qtok = Counter(_tokenize(query))
    scored = []
    for doc in corpus:
        dtok = Counter(_tokenize(doc["text"]))
        overlap = sum(min(qtok[w], dtok[w]) for w in qtok)
        norm = max(1, sum(dtok.values()))
        scored.append({"id": doc["id"], "text": doc["text"], "score": overlap / norm})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

def heuristic_grounding(answer: str, ctx_docs: List[str]) -> float:
    a = _tokenize(answer)
    if not a:
        return 0.0
    ctx = Counter(_tokenize("\n".join(ctx_docs)))
    hit = sum(1 for w in a if ctx[w] > 0)
    return round(hit / len(a), 3)

# ------------------------ prompt budgeting ---------------------------
def _maybe_get_hf_tokenizer_for_synth(cfg: Dict[str, Any], models: Dict[str, Any]):
    """Return HF tokenizer if synth backend is HF, else None."""
    handle = cfg["llms"]["synthesize"]["model"]
    entry = models.get(handle, {})
    if entry.get("backend", "hf") != "hf":
        return None, None
    model_id = entry["id"]
    tok = _HF_CACHE.get(("tok", model_id))
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_id)
        _HF_CACHE[("tok", model_id)] = tok
    # try to infer model max length (default GPT2 family ~1024)
    max_len = getattr(tok, "model_max_length", None)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > 1_000_000:
        max_len = 1024
    return tok, max_len

def _trim_to_tokens(text: str, tok, limit: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    ids = ids[:limit]
    return tok.decode(ids)

def _pack_prompt(question: str,
                 ctx_docs: List[str],
                 prompt_head: str,
                 cite_note: str,
                 tok: Optional[Any],
                 total_budget: int,
                 per_doc_budget: int) -> str:
    """
    Assemble a prompt within a token budget.
    Strategy:
      1) Reserve ~20% for the question/instructions, spend the rest on context.
      2) Trim each doc to per_doc_budget.
      3) If still over budget, drop the worst (last) docs until within budget.
    """
    if tok is None:
        # Char-based fallback (ollama or unknown): approximate budgets by chars (~4 chars/token)
        char_budget = total_budget * 4
        per_doc_chars = per_doc_budget * 4
        header = f"{prompt_head}\nQUESTION: {question}\n\nEVIDENCE:\n"
        pieces = []
        for d in ctx_docs:
            pieces.append("- " + (d[:per_doc_chars]))
        prompt = header + "\n".join(pieces) + f"\n\n{cite_note}\n"
        return prompt[:char_budget]

    # Token-accurate budgeting
    header = f"{prompt_head}\nQUESTION: {question}\n\nEVIDENCE:\n"
    header = _trim_to_tokens(header, tok, int(0.2 * total_budget))
    items = []
    for d in ctx_docs:
        items.append("- " + _trim_to_tokens(d, tok, per_doc_budget))
    body = "\n".join(items)
    rest_budget = max(1, total_budget - len(tok.encode(header)) - 50)  # keep a small tail for cite_note
    # If body too long, drop from end until fits
    while len(tok.encode(body)) > rest_budget and items:
        items.pop()              # drop last doc
        body = "\n".join(items)
    cite = _trim_to_tokens(cite_note, tok, 48)
    prompt = header + body + "\n\n" + cite + "\n"
    # Final hard trim if still slightly over
    if len(tok.encode(prompt)) > total_budget:
        prompt = _trim_to_tokens(prompt, tok, total_budget)
    return prompt

# ------------------------ pipeline entry ------------------------------
def run(cfg: Dict[str, Any],
        models: Dict[str, Any],
        trace: Trace,
        metrics: MetricLog,
        corpus: List[Dict[str, str]],
        queries: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline with prompt budgeting.
    """
    top_k = int(cfg["retriever"].get("top_k", 3))
    synth = build_stage_runner(cfg["llms"]["synthesize"], models)

    # Prompt budgets (safe defaults for GPT-2/DistilGPT2)
    prompt_cfg = cfg.get("prompt", {})
    tok, model_max_len = _maybe_get_hf_tokenizer_for_synth(cfg, models)
    # Keep generation headroom (e.g., 160 new tokens) by shrinking input budget
    max_new = int(cfg["llms"]["synthesize"].get("params", {}).get("max_new_tokens", 160))
    # Total context budget cannot exceed model_max_len - max_new - a small safety margin
    hard_cap = max(256, model_max_len - max_new - 16)
    total_budget = int(prompt_cfg.get("budget_tokens", min(900, hard_cap)))
    per_doc_budget = int(prompt_cfg.get("ctx_per_doc_tokens", 300))

    results: Dict[str, Any] = {}

    for q in queries:
        qid, qtext = q["id"], q["text"]

        with trace.span("retrieve") as sp:
            top = retrieve_simple_overlap(qtext, corpus, k=top_k)
            sp.log({"k": len(top), "scores": [r["score"] for r in top]})
            ctx_docs = [r["text"] for r in top]

        with trace.span("synthesize") as sp:
            prompt = _pack_prompt(
                question=qtext,
                ctx_docs=ctx_docs,
                prompt_head="Answer the question using ONLY the evidence.",
                cite_note="Answer briefly and cite by doc_ids if needed.",
                tok=tok,
                total_budget=total_budget,
                per_doc_budget=per_doc_budget,
            )
            out = synth(prompt)
            trace.add_tokens(out["usage"].get("prompt", 0), out["usage"].get("completion", 0))
            sp.log({"latency_s": out.get("latency_s", None)})
            answer = out["text"]

        with trace.span("judge_heuristic") as sp:
            grounding = heuristic_grounding(answer, ctx_docs)
            sp.log({"grounding": grounding})

        metrics.add(pipeline="rag", item=qid, relevance=grounding,
                    faithfulness=grounding, latency_ms=trace.wall_time_ms, cost_usd=0.0)

        results[qid] = {"answer": answer, "grounding": grounding, "ctx_ids": [r["id"] for r in top]}

    return {"results": results}