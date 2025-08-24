# __init__.py — stub
# xpipe/runners/__init__.py
import os

def backend_kind(cfg):
    return (cfg.get("backend") or os.getenv("XPIPE_BACKEND","hf_local")).lower()

def openai_compat_client():
    # Reuse your existing openai package; point it at any OpenAI-compatible server.
    from openai import OpenAI
    base_url = os.getenv("XPIPE_BASE_URL") or "http://localhost:11434/v1"  # Ollama default
    api_key  = os.getenv("XPIPE_API_KEY") or "dummy"
    return OpenAI(base_url=base_url, api_key=api_key)