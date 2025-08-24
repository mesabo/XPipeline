# xpipe/runners/rag.py
# ======================================================================
# RAG runner with:
#   • pluggable retriever: simple_overlap | jaccard
#   • optional judge (enabled via llms.judge.enabled)
#   • prompt budgeting to avoid small-context model crashes
#   • logs per-stage tokens, latency, and quality metrics
# ======================================================================

from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter
import time, os, json

from xpipe.metrics import MetricLog, rouge_l_fscore, f1_token_score

# --------------------------- tokenization -----------------------------
def _tok(s: str) -> List[str]:
    return [t.lower() for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s).split()]

# ---------------------------- retrieval -------------------------------
def _retrieve_simple(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    q = Counter(_tok(query))
    sc = []
    for d in corpus:
        dt = Counter(_tok(d["text"]))
        overlap = sum(min(q[w], dt[w]) for w in q)
        norm = max(1, sum(dt.values()))
        sc.append({"id": d["id"], "text": d["text"], "score": overlap / norm})
    sc.sort(key=lambda x: x["score"], reverse=True)
    return sc[:k]

def _retrieve_jaccard(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    q = set(_tok(query))
    sc = []
    for d in corpus:
        s = set(_tok(d["text"]))
        inter, union = len(q & s), max(1, len(q | s))
        sc.append({"id": d["id"], "text": d["text"], "score": inter / union})
    sc.sort(key=lambda x: x["score"], reverse=True)
    return sc[:k]

RETRIEVERS = {"simple_overlap": _retrieve_simple, "jaccard": _retrieve_jaccard}

def _grounding(answer: str, ctx_docs: List[str]) -> float:
    a = _tok(answer)
    if not a: return 0.0
    ctx = Counter(_tok("\n".join(ctx_docs)))
    hit = sum(1 for w in a if ctx[w] > 0)
    return round(hit / len(a), 6)

# ----------------------------- backends -------------------------------
_HF = {}

def _hf_gen(model_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    mdl = _HF.get(("m", model_id)); tok = _HF.get(("t", model_id))
    if mdl is None or tok is None:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        mdl.eval()
        if torch.cuda.is_available(): mdl.to("cuda")
        _HF[("m", model_id)] = mdl; _HF[("t", model_id)] = tok
    ids = tok(prompt, return_tensors="pt")
    if mdl.device.type == "cuda": ids = {k: v.to("cuda") for k, v in ids.items()}
    kwargs = dict(
        max_new_tokens=int(params.get("max_new_tokens", 160)),
        do_sample=bool(params.get("do_sample", params.get("temperature", 0.0) > 0)),
        temperature=float(params.get("temperature", 0.0)),
        top_p=float(params.get("top_p", 1.0)),
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
    )
    t0 = time.time()
    with torch.no_grad():
        out = mdl.generate(**ids, **kwargs)
    latency = time.time() - t0
    full = tok.decode(out[0], skip_special_tokens=True)
    prompt_text = tok.decode(ids["input_ids"][0], skip_special_tokens=True)
    completion = full[len(prompt_text):] if full.startswith(prompt_text) else full
    return {"text": completion.strip(),
            "usage": {"prompt": len(tok.encode(prompt)), "completion": len(tok.encode(completion))},
            "latency_s": latency}

def _ollama_gen(model_id: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
    r = requests.post(f"{base}/api/generate",
                      json={"model": model_id, "prompt": prompt, "stream": False,
                            "options": {"temperature": float(params.get("temperature", 0.2)),
                                        "top_p": float(params.get("top_p", 0.9)),
                                        "num_predict": int(params.get("max_new_tokens", 200))}},
                      timeout=600)
    r.raise_for_status()
    j = r.json()
    return {"text": j.get("response", "").strip(),
            "usage": {"prompt": j.get("prompt_eval_count", 0), "completion": j.get("eval_count", 0)},
            "latency_s": j.get("total_duration", 0)/1e9 if "total_duration" in j else None}

def build_runner(stage_cfg: Dict[str, Any], models: Dict[str, Any]):
    handle = stage_cfg["model"]
    if handle in models:
        entry = models[handle]; backend = entry.get("backend", "hf"); model_id = entry["id"]; base = entry.get("params", {})
    else:
        if "/" not in handle: raise ValueError(f"Unknown model handle '{handle}'")
        backend, model_id = handle.split("/", 1); base = {}
    overrides = stage_cfg.get("params", {})
    def _run(prompt: str, **kw):
        p = {**base, **overrides, **kw}
        return _hf_gen(model_id, prompt, p) if backend == "hf" else _ollama_gen(model_id, prompt, p)
    _run._backend = backend; _run._model_id = model_id
    return _run

# ----------------------------- helpers --------------------------------
def _load_jsonl_corpus(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            out.append({"id": str(o.get("id", len(out))), "text": str(o.get("text", ""))})
    if not out: raise ValueError(f"Empty corpus at {path}")
    return out

def _clamp_prompt(question: str, docs: List[str], budget: int, per_doc: int) -> List[str]:
    # 4 chars ~ 1 token heuristic
    per_chars = max(1, per_doc*4); total_chars = max(1, budget*4)
    docs = [d if len(d) <= per_chars else d[:per_chars] for d in docs]
    head = f"Answer the question using ONLY the evidence.\nQUESTION: {question}\n\nEVIDENCE:\n- "
    base = head + "\n- ".join(docs) + "\n\nAnswer:"
    while len(base) > total_chars and len(docs) > 1:
        docs.pop()
        base = head + "\n- ".join(docs) + "\n\nAnswer:"
    return docs

# ------------------------------ pipeline ------------------------------
def run(cfg: Dict[str, Any], models: Dict[str, Any], metrics: MetricLog,
        corpus: List[Dict[str, str]], queries: List[Dict[str, Any]]) -> Dict[str, Any]:

    retr_name = (cfg.get("retriever", {}) or {}).get("name", "simple_overlap")
    top_k     = int((cfg.get("retriever", {}) or {}).get("top_k", 3))
    retrieve  = RETRIEVERS.get(retr_name)
    if retrieve is None:
        raise ValueError(f"Unknown retriever '{retr_name}'")

    synth = build_runner(cfg["llms"]["synthesize"], models)

    judge_cfg = (cfg.get("llms", {}) or {}).get("judge", {})
    judge_enabled = bool(judge_cfg.get("enabled", True))
    judge_runner  = build_runner(judge_cfg, models) if (judge_enabled and "model" in judge_cfg) else None

    prompt_cfg = cfg.get("prompt", {}) or {}
    budget_tokens   = int(prompt_cfg.get("budget_tokens", 900))
    ctx_per_doc_tok = int(prompt_cfg.get("ctx_per_doc_tokens", 300))

    results = {}
    for q in queries:
        qid, qtext, ref = str(q["id"]), q["text"], q.get("ref", "")

        t0 = time.time()
        top = retrieve(qtext, corpus, k=top_k)
        retr_latency_ms = (time.time() - t0) * 1000.0
        ctx_docs = _clamp_prompt(qtext, [r["text"] for r in top], budget_tokens, ctx_per_doc_tok)

        prompt = ("Answer the question using ONLY the evidence.\n"
                  f"QUESTION: {qtext}\n\nEVIDENCE:\n- " + "\n- ".join(ctx_docs) + "\n\nAnswer:")

        out_s = synth(prompt)
        ans = out_s["text"]
        synth_pt = int(out_s.get("usage", {}).get("prompt", 0))
        synth_ct = int(out_s.get("usage", {}).get("completion", 0))
        synth_lat_ms = float(out_s.get("latency_s", 0.0)) * 1000.0

        grounding = _grounding(ans, ctx_docs)

        judge_pt = judge_ct = 0
        judge_lat_ms = 0.0
        if judge_runner is not None:
            j_prompt = ("Does the ANSWER strictly stay within the EVIDENCE facts? Reply with OK or DRIFT.\n\n"
                        "EVIDENCE:\n- " + "\n- ".join(ctx_docs[:3]) + "\n\nANSWER:\n" + ans[:1200])
            out_j = judge_runner(j_prompt, max_new_tokens=16, temperature=0.0)
            judge_pt = int(out_j.get("usage", {}).get("prompt", 0))
            judge_ct = int(out_j.get("usage", {}).get("completion", 0))
            judge_lat_ms = float(out_j.get("latency_s", 0.0)) * 1000.0

        rouge_f = f1_t = ""
        if ref:
            try:
                rouge_f = round(rouge_l_fscore(ans, ref), 6)
                f1_t    = round(f1_token_score(ans, ref), 6)
            except Exception:
                rouge_f = f1_t = ""

        metrics.add(
            pipeline="rag",
            item=qid,
            retriever=retr_name,
            judge_enabled=bool(judge_runner is not None),
            relevance=grounding,
            rougeL_f=rouge_f,
            f1_token=f1_t,
            latency_ms=round(retr_latency_ms + synth_lat_ms + judge_lat_ms, 2),
            cost_usd=0.0,
            synth_prompt_tokens=synth_pt,
            synth_completion_tokens=synth_ct,
            judge_prompt_tokens=judge_pt,
            judge_completion_tokens=judge_ct,
            synth_model=getattr(synth, "_model_id", ""),
            judge_model=getattr(judge_runner, "_model_id", "") if judge_runner else "",
        )

        results[qid] = {"answer": ans, "grounding": grounding, "ctx_ids": [r["id"] for r in top]}

    return {"results": results}