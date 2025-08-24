# rag.py — stub
# xpipe/runners/rag.py
# ----------------------------------------------------------------------
# XPipe RAG Runner (Hybrid Retrieval + Generation + Judging)
# ----------------------------------------------------------------------
# Purpose
#   Minimal but practical RAG pipeline composed of:
#     • Router/Retrieval: SentenceTransformer dense retrieval (+ optional CrossEncoder re‑rank).
#     • Synthesis: either Hugging Face causal LM (local) or OpenAI‑compatible client.
#     • Judging: free, local judge using NLI (entailment) and CrossEncoder relevance.
#
# Backends
#   - HF-local (free): transformers AutoModelForCausalLM + AutoTokenizer.
#   - OpenAI‑compatible: any provider exposing /v1/chat/completions (optional).
#
# Notes
#   - This runner expects small/medium free models when running locally.
#   - All external services (OpenAI‑compatible) are optional; HF‑only works offline.
#   - Imports `tracer` from xpipe.trace; ensure a global tracer is initialized upstream.
#
# I/O Contract
#   run(question, doc_chunks, cfg, metrics) -> dict with:
#       {"answer": str,
#        "selected_indices": List[int],
#        "faithfulness": float,
#        "answer_relevance": float,
#        "retrieval_relevance": float}
#
# Metrics
#   Logs a composite score into `metrics`:
#       0.5 * faithfulness + 0.25 * answer_relevance + 0.25 * retrieval_relevance
# ----------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from xpipe.metrics import MetricLog
from xpipe.trace import tracer
from . import backend_kind, openai_compat_client

# ---------- HF-local (free) generator ----------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class RAGModelsHF:
    """Container for HF-local models used by this RAG runner."""
    embedder: SentenceTransformer
    cross: Optional[CrossEncoder]
    tok: AutoTokenizer
    lm: AutoModelForCausalLM


@dataclass
class RAGModelsOA:
    """Container for OpenAI-compatible stack used by this RAG runner."""
    embedder: SentenceTransformer
    cross: Optional[CrossEncoder]
    client: Any
    model_name: str


def _load_shared(cfg):
    """
    Load shared retrieval models (embedder + optional cross encoder).

    cfg["router"]:
        - embedder: str  (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        - cross_encoder: Optional[str] (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
    """
    emb = SentenceTransformer(cfg["router"]["embedder"])
    cross = CrossEncoder(cfg["router"].get("cross_encoder")) if cfg["router"].get("cross_encoder") else None
    return emb, cross


def load_models(cfg):
    """
    Load per-backend synthesis models while reusing shared retrieval blocks.

    Returns
    -------
    tuple[str, RAGModelsHF|RAGModelsOA]
        ("hf", RAGModelsHF(...)) or ("oa", RAGModelsOA(...))
    """
    kind = backend_kind(cfg)
    emb, cross = _load_shared(cfg)
    if kind == "hf_local":
        name = cfg["synthesizer"]["model"]
        tok = AutoTokenizer.from_pretrained(name)
        lm  = AutoModelForCausalLM.from_pretrained(
            name,
            device_map=cfg["synthesizer"].get("device_map","auto"),
            torch_dtype=cfg["synthesizer"].get("dtype","auto"),
        )
        return ("hf", RAGModelsHF(emb, cross, tok, lm))
    else:
        client = openai_compat_client()
        return ("oa", RAGModelsOA(emb, cross, client, cfg["synthesizer"]["model"]))


# ---------- retrieval ----------
def score_chunks(question: str, chunks: List[str], emb_model, cross=None, top_k=5):
    """
    Rank context chunks for a question using bi-encoder similarity, with optional cross-encoder re-ranking.

    Parameters
    ----------
    question : str
        User question / query.
    chunks : List[str]
        Candidate passages.
    emb_model : SentenceTransformer
        Bi-encoder for dense retrieval (cosine on normalized embeddings).
    cross : Optional[CrossEncoder]
        If provided, re-ranks the top candidates from the bi-encoder stage.
    top_k : int
        Final number of chunks to return.

    Returns
    -------
    List[int]
        Indices of selected chunks (length == top_k).
    """
    qv = emb_model.encode([question], normalize_embeddings=True)
    cv = emb_model.encode(chunks, normalize_embeddings=True)
    sims = np.asarray(cv @ qv.T).squeeze(-1)
    idx = np.argsort(-sims)[: max(top_k*3, top_k)]
    if cross:
        pairs = [[question, chunks[i]] for i in idx]
        rer = cross.predict(pairs)
        rer_idx = np.argsort(-rer)[:top_k]
        return [idx[i] for i in rer_idx]
    return idx[:top_k].tolist()


# ---------- generation ----------
def _gen_hf(prompt: str, tok, lm, max_new_tokens, temperature) -> str:
    """
    Generate with Hugging Face CausalLM (local).

    Notes
    -----
    - do_sample is toggled by temperature > 0.
    - Uses EOS token as pad_token_id to avoid warnings.
    """
    toks = tok(prompt, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        out = lm.generate(**toks,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def _gen_oa(prompt: str, client, model_name, max_new_tokens, temperature) -> str:
    """
    Generate with an OpenAI‑compatible /v1/chat/completions API client.

    Expects
    -------
    client.chat.completions.create(model=..., messages=[...], temperature=..., max_tokens=...)
    """
    # OpenAI-compatible /v1/chat/completions
    r = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=float(temperature),
        max_tokens=int(max_new_tokens)
    )
    return r.choices[0].message.content


# ---------- judge (HF-local only, all free) ----------
from transformers import AutoModelForSequenceClassification, AutoTokenizer as NliTok

class Judge:
    """
    Free/local judge combining NLI (entailment) for faithfulness and CrossEncoder for relevance.

    Components
    ----------
    - NLI model (sequence classification) for answer faithfulness vs. sources.
    - Cross-encoder for:
        * answer relevance to question
        * retrieval relevance (best context vs. question)
    """

    def __init__(self, nli_name: str, cross_name: str):
        self.nli_tok = NliTok.from_pretrained(nli_name)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_name).eval()
        self.ce = CrossEncoder(cross_name)

    @torch.no_grad()
    def faithfulness(self, answer: str, sources: List[str]) -> float:
        """
        Estimate faithfulness via NLI entailment probability.

        Returns
        -------
        float
            P(entailment) in [0,1] for (premise=concatenated sources, hypothesis=answer).
        """
        premise = "\n".join(sources)[:4000]
        t = self.nli_tok(premise, answer, return_tensors="pt", truncation=True)
        logits = self.nli(**t).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return float(probs[2])

    def relevance(self, question: str, text: str) -> float:
        """
        Cross-encoder score for (question, text). Higher is more relevant.
        """
        return float(self.ce.predict([[question, text]])[0])


# ---------- pipeline entry ----------
def run(question: str, doc_chunks: List[str], cfg, metrics: MetricLog) -> Dict[str, Any]:
    """
    Execute RAG pipeline over a single question.

    Steps
    -----
    1) Retrieval: dense encode + (optional) cross re-rank to pick top_k chunks.
    2) Synthesis: HF-local or OpenAI-compatible generation.
    3) Judging: NLI faithfulness + CE relevance (answer and retrieval).

    Side Effects
    ------------
    - Logs a tracing span "route" via global `tracer`.
    - Appends a composite score row to `metrics`.

    Returns
    -------
    Dict[str, Any]
        {
          "answer": str,
          "selected_indices": List[int],
          "faithfulness": float,
          "answer_relevance": float,
          "retrieval_relevance": float
        }
    """
    kind, models = load_models(cfg)
    judge = Judge(cfg["judge"]["nli_model"], cfg["judge"]["cross_encoder"])

    with tracer.span("route"):
        top_idx = score_chunks(question, doc_chunks, models.embedder, models.cross, top_k=5)

    ctx = [doc_chunks[i] for i in top_idx]
    prompt = (
        f"Question: {question}\n\nUse only these sources:\n\n" +
        "\n\n---\n\n".join([f"[{i}] {c}" for i, c in enumerate(ctx)]) +
        "\n\nAnswer briefly and cite [idx] like [0],[1]."
    )

    gen_cfg = cfg["generation"]
    if kind == "hf":
        ans = _gen_hf(prompt, models.tok, models.lm, gen_cfg["max_new_tokens"], gen_cfg["temperature"])
    else:
        ans = _gen_oa(prompt, models.client, models.model_name, gen_cfg["max_new_tokens"], gen_cfg["temperature"])

    faith = judge.faithfulness(ans, ctx)
    ans_rel = judge.relevance(question, ans)
    ret_rel = max(judge.relevance(question, c) for c in ctx)

    metrics.log_row(step="rag", ok=True,
                    score=float(0.5*faith + 0.25*ans_rel + 0.25*ret_rel),
                    extras={"faithfulness": faith, "answer_rel": ans_rel, "retrieval_rel": ret_rel})

    return {"answer": ans, "selected_indices": top_idx,
            "faithfulness": faith, "answer_relevance": ans_rel, "retrieval_relevance": ret_rel}