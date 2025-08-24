# xpipe/llm_backends.py
# SPDX-License-Identifier: BSD-3-Clause
"""
LLM Backends — Unified interface for local model inference.

This module provides wrappers for running free, local large language models
via either Hugging Face Transformers or an Ollama server. It is dependency-light,
self-contained, and designed to be easily extended with new backends.

Features
--------
- Hugging Face backend (`hf_generate`):
  Runs small/medium causal models locally (e.g. GPT-2, DistilGPT2, TinyLlama,
  Qwen Instruct). Uses PyTorch + transformers.

- Ollama backend (`ollama_generate`):
  Sends HTTP requests to a locally running Ollama server
  (default: http://localhost:11434). Supports community models such as
  Llama 3.2, Qwen 2.5, DeepSeek R1, etc.

- Unified dispatcher (`call_llm`):
  Single entrypoint that dispatches to the appropriate backend
  based on the `backend` string ("hf" or "ollama").

Design
------
- Hugging Face models are cached in memory per model_id for efficiency.
- Ollama requests are blocking and return structured JSON.
- Both backends return a standard dictionary format:
    {
      "text": <generated string>,
      "usage": {"prompt": <int>, "completion": <int>},
      "latency_s": <float seconds>
    }
- No API keys or paid cloud services are required.
"""

from __future__ import annotations
import os, time, json
from typing import Dict, Any

# ---- Hugging Face Transformers (local) ----
_HF_CACHE: Dict[str, Any] = {}


def hf_generate(model_id: str, prompt: str, **kw) -> Dict[str, Any]:
    """
    Generate text using a Hugging Face CausalLM model.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier (e.g. "gpt2", "distilgpt2",
        "Qwen/Qwen2.5-0.5B-Instruct").
    prompt : str
        Input text prompt.
    **kw : dict
        Generation hyperparameters:
          - max_new_tokens (int): maximum number of tokens to generate (default 128)
          - temperature (float): sampling temperature (default 0.7)
          - top_p (float): nucleus sampling cutoff (default 0.95)

    Returns
    -------
    dict
        {
          "text": generated completion string,
          "usage": {"prompt": <int>, "completion": <int>},
          "latency_s": wall-clock runtime in seconds
        }

    Notes
    -----
    - Models are cached globally to avoid reloading weights each call.
    - If CUDA is available, the model and tensors are moved to GPU.
    - This function is intended for small / medium models that can run on local hardware.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    entry = _HF_CACHE.get(model_id)
    if entry is None:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )
        mdl.eval()
        if torch.cuda.is_available():
            mdl = mdl.to("cuda")
        _HF_CACHE[model_id] = entry = {"tok": tok, "mdl": mdl}
    tok, mdl = entry["tok"], entry["mdl"]

    max_new_tokens = int(kw.get("max_new_tokens", 128))
    temperature     = float(kw.get("temperature", 0.7))
    top_p           = float(kw.get("top_p", 0.95))

    inputs = tok(prompt, return_tensors="pt")
    if mdl.device.type == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id
        )
    dt = time.time() - t0

    text = tok.decode(out_ids[0], skip_special_tokens=True)
    completion = text[len(prompt):] if text.startswith(prompt) else text

    # Estimate token usage (no billing, just observability)
    ptok = len(tok.encode(prompt))
    ctok = len(tok.encode(completion))
    return {"text": completion, "usage": {"prompt": ptok, "completion": ctok}, "latency_s": dt}


# ---- Ollama (local server) ----
def ollama_generate(model_id: str, prompt: str, **kw) -> Dict[str, Any]:
    """
    Generate text using an Ollama model via HTTP API.

    Parameters
    ----------
    model_id : str
        Ollama model identifier (e.g. "llama3.2:3b-instruct", "qwen2.5:3b-instruct").
    prompt : str
        Input text prompt.
    **kw : dict
        Generation hyperparameters:
          - max_new_tokens (int): number of tokens to generate (default 128)
          - temperature (float): sampling temperature (default 0.7)
          - top_p (float): nucleus sampling cutoff (default 0.95)

    Returns
    -------
    dict
        {
          "text": generated completion string,
          "usage": {"prompt": <int>, "completion": <int>},
          "latency_s": wall-clock runtime in seconds
        }

    Notes
    -----
    - Requires `ollama serve` to be running locally.
    - Default server: http://localhost:11434
    - Ollama must have the model pulled in advance (e.g., `ollama pull llama3.2:3b-instruct`).
    """
    import requests
    base = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
    url  = f"{base}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(kw.get("temperature", 0.7)),
            "top_p": float(kw.get("top_p", 0.95)),
            "num_predict": int(kw.get("max_new_tokens", 128))
        }
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    dt = time.time() - t0
    r.raise_for_status()
    data = r.json()
    txt = data.get("response", "")
    usage = {
        "prompt": data.get("prompt_eval_count", 0),
        "completion": data.get("eval_count", 0)
    }
    return {"text": txt, "usage": usage, "latency_s": dt}


# ---- Unified dispatcher ----
BACKENDS = {
    "hf": hf_generate,        # Hugging Face Transformers (local CPU/GPU)
    "ollama": ollama_generate # Ollama server (local HTTP API)
}


def call_llm(backend: str, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Unified LLM dispatcher.

    Parameters
    ----------
    backend : str
        Either "hf" (Hugging Face) or "ollama".
    model_id : str
        The model identifier (HF hub id or Ollama tag).
    prompt : str
        Input text to generate from.
    **kwargs : dict
        Backend-specific generation parameters.

    Returns
    -------
    dict
        Standardized response dictionary (see backend docstrings).
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(BACKENDS)}")
    return BACKENDS[backend](model_id, prompt, **kwargs)