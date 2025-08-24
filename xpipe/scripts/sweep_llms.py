#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, subprocess, yaml, itertools, re, socket
from typing import Dict, Any, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../xpipe/scripts -> .../xpipe
REPO = os.path.dirname(ROOT)
CFG_DIR = os.path.join(ROOT, "configs")
TMP_DIR = os.path.join(REPO, "output", "xpipe", "tmp_configs")
EXP_PATH = os.path.join(CFG_DIR, "experiment_rag.yaml")
MODELS_PATH = os.path.join(CFG_DIR, "models.yaml")

def load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def sanitize(s: str) -> str:
    s = s.lower().replace("/", "_").replace(":", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def list_handles(models_yaml: Dict[str, Any], backends: List[str]) -> List[Tuple[str,str]]:
    """
    Return [(backend, model_id), ...] from your models.yaml.
    Expect either:
      A) registry style (our earlier version):
         hf/distilgpt2: {backend: hf, id: distilgpt2, ...}
         ollama/llama3.2-3b: {backend: ollama, id: llama3.2:3b, ...}
      B) grouped style (optional):
         backends:
           hf: {enabled: true, models: [gpt2, distilgpt2, ...]}
           ollama: {enabled: false, models: [...]}
    """
    out: List[Tuple[str,str]] = []

    if "backends" in models_yaml:  # grouped style
        for bk, spec in models_yaml["backends"].items():
            if bk not in backends: continue
            if not spec.get("enabled", True): continue
            for mid in spec.get("models", []):
                out.append((bk, mid))
        return out

    # registry style
    for handle, spec in models_yaml.items():
        if not isinstance(spec, dict): continue
        bk = spec.get("backend") or spec.get("provider")
        mid = spec.get("id") or spec.get("model") or handle
        if not bk or not mid: continue
        if bk in backends:
            out.append((bk, mid))
    return out

def build_cfg_name(base_name: str, synth: Tuple[str,str], judge: Tuple[str,str] | None, mode: str) -> str:
    sb, sm = synth
    name = f"{base_name}_syn_{sanitize(sb)}_{sanitize(sm)}"
    if mode == "grid" and judge:
        jb, jm = judge
        name += f"_jdg_{sanitize(jb)}_{sanitize(jm)}"
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth","judge","grid"], default="synth")
    ap.add_argument("--include", nargs="*", default=None, help='substring filter, e.g. "hf/" or "qwen"')
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backends", default="hf,ollama")  # auto pruned if ollama unavailable by wrapper
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(EXP_PATH) or not os.path.exists(MODELS_PATH):
        print("[sweep] ERROR: missing xpipe/configs/experiment_rag.yaml or models.yaml", file=sys.stderr)
        sys.exit(2)

    base = load_yaml(EXP_PATH)
    models_yaml = load_yaml(MODELS_PATH)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    handles = list_handles(models_yaml, backends)

    # Apply include/exclude filters
    def keep(h: Tuple[str,str]) -> bool:
        b, m = h
        t = f"{b}/{m}"
        if args.include and not any(s in t for s in args.include): return False
        if args.exclude and any(s in t for s in args.exclude): return False
        return True

    handles = [h for h in handles if keep(h)]

    # Build run list
    runs: List[Tuple[str, Dict[str,Any]]] = []
    if args.mode == "synth":
        for syn in handles:
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["name"] = build_cfg_name(base["name"], syn, None, args.mode)
            cfg["models_path"] = MODELS_PATH
            # Maintain your existing schema: llms.synthesize.model is a "handle" if registry style;
            # otherwise store backend/id in the stage "select" map.
            if "backends" in models_yaml:  # grouped style
                bk, mid = syn
                cfg.setdefault("llms", {}).setdefault("synthesize", {})["select"] = {"backend": bk, "id": mid}
            else:  # registry style: need a matching key. We fabricate a temporary selectable key:
                bk, mid = syn
                cfg.setdefault("llms", {}).setdefault("synthesize", {})["select"] = {"backend": bk, "id": mid}
            runs.append((cfg["name"], cfg))

    elif args.mode == "judge":
        for jdg in handles:
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["name"] = build_cfg_name(base["name"], ("_keep_", "_keep_"), jdg, args.mode)
            cfg["models_path"] = MODELS_PATH
            bk, mid = jdg
            cfg.setdefault("llms", {}).setdefault("judge", {})["select"] = {"backend": bk, "id": mid}
            runs.append((cfg["name"], cfg))

    else:  # grid
        for syn, jdg in itertools.product(handles, handles):
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["name"] = build_cfg_name(base["name"], syn, jdg, args.mode)
            cfg["models_path"] = MODELS_PATH
            (sb, sm), (jb, jm) = syn, jdg
            cfg.setdefault("llms", {}).setdefault("synthesize", {})["select"] = {"backend": sb, "id": sm}
            cfg.setdefault("llms", {}).setdefault("judge", {})["select"] = {"backend": jb, "id": jm}
            runs.append((cfg["name"], cfg))

    total = len(runs)
    if args.limit and args.limit > 0:
        runs = runs[: args.limit]

    print(f"[sweep] Handles discovered: {len(handles)}")
    print(f"[sweep] Mode={args.mode}  -> planned runs={total}  -> after limit={len(runs)}")
    print(f"[sweep] Writing configs to: {TMP_DIR}")
    os.makedirs(TMP_DIR, exist_ok=True)

    # DRY-RUN: only report counts, do not write or execute
    if args.dry_run:
        for nm, _ in runs[:10]:
            print(f"  - {nm}.yaml")
        if len(runs) > 10:
            print(f"  ... +{len(runs)-10} more")
        return

    # Write temp configs and execute
    failures = 0
    for name, cfg in runs:
        tmp_cfg = os.path.join(TMP_DIR, f"{name}.yaml")
        save_yaml(cfg, tmp_cfg)
        print(f"→ {sys.executable} xpipe/main.py --config {tmp_cfg}")
        rc = subprocess.call([sys.executable, os.path.join("xpipe","main.py"), "--config", tmp_cfg])
        if rc != 0:
            print(f"[sweep] FAILED: {tmp_cfg} (rc={rc})")
            failures += 1

    print(f"[sweep] Done. total={len(runs)}, failed={failures}")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()