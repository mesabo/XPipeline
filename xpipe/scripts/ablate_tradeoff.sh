#!/usr/bin/env bash
# ----------------------------------------------------------------------
# ablate_tradeoff.sh  (robust)
# Compares:
#   retriever ∈ {simple_overlap, jaccard}
#   judge     ∈ {on, off}
#
# Usage:
#   bash xpipe/scripts/ablate_tradeoff.sh [CONFIG_PATH]
# Defaults to xpipe/configs/experiment_rag.yaml
# ----------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-xpipe/configs/experiment_rag.yaml}"
PY="${PYTHON_BIN:-python}"

# Ensure paths are right for THIS repo
MAIN_PY="xpipe/main.py"

tmpdir="$(mktemp -d)"
cleanup(){ rm -rf "$tmpdir"; }
trap cleanup EXIT

# Use Python to copy+modify YAML safely
make_cfg_py () {
"$PY" - "$@" <<'PYCODE'
import sys, yaml, os
base_path, out_path, run_name, retriever_name, judge_enabled = sys.argv[1:6]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Ensure required blocks exist
cfg["name"] = run_name
cfg.setdefault("retriever", {})
cfg["retriever"]["name"] = retriever_name
# Keep top_k untouched if present
cfg.setdefault("llms", {})
cfg["llms"].setdefault("judge", {})
cfg["llms"]["judge"]["enabled"] = (judge_enabled.lower() == "true")

# Auto outputs so we don't overwrite same file
cfg.setdefault("outputs", {})
cfg["outputs"]["run_jsonl"] = ""     # let main auto-name by run name & timestamp
cfg["outputs"]["metrics_csv"] = ""   # idem

# Write updated config
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out_path)
PYCODE
}

run_one () {
  local cfg="$1"
  echo "→ running: $cfg"
  "$PY" "$MAIN_PY" --config "$cfg" 1>/dev/null
}

summarize () {
"$PY" - <<'PY'
import glob, os, pandas as pd
met_dir = "output/xpipe/metrics"
rows=[]
for f in sorted(glob.glob(os.path.join(met_dir, "*.csv"))):
    run = os.path.basename(f).replace(".csv","")
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    # infer tag fields if present
    def col(c): return df[c].iloc[0] if (c in df and len(df)>0) else None
    rows.append(dict(
        run=run,
        retriever=col("retriever"),
        judge=col("judge_enabled"),
        mean_ground = df["relevance"].mean()       if "relevance" in df else float("nan"),
        mean_rougeL= df["rougeL_f"].mean()        if "rougeL_f" in df else float("nan"),
        mean_f1    = df["f1_token"].mean()        if "f1_token" in df else float("nan"),
        mean_latency_ms = df["latency_ms"].mean() if "latency_ms" in df else float("nan"),
        synth_tokens = (df["synth_prompt_tokens"].sum() if "synth_prompt_tokens" in df else 0)
                      + (df["synth_completion_tokens"].sum() if "synth_completion_tokens" in df else 0),
        judge_tokens = (df["judge_prompt_tokens"].sum() if "judge_prompt_tokens" in df else 0)
                      + (df["judge_completion_tokens"].sum() if "judge_completion_tokens" in df else 0),
    ))
out = pd.DataFrame(rows)
# Show only the four tradeoff runs if repository has other CSVs
mask = out["run"].str.contains("tradeoff_", na=False)
if mask.any():
    out = out[mask]
if out.empty:
    print("[no matching runs found in output/xpipe/metrics]")
else:
    # Sort for readability
    if "retriever" in out.columns and "judge" in out.columns:
        out = out.sort_values(["retriever","judge","run"])
    print(out.to_string(index=False))
PY
}

# Build 4 configs cleanly
c1="$tmpdir/tradeoff_simple_on.yaml"
c2="$tmpdir/tradeoff_simple_off.yaml"
c3="$tmpdir/tradeoff_jaccard_on.yaml"
c4="$tmpdir/tradeoff_jaccard_off.yaml"

make_cfg_py "$CONFIG" "$c1" "tradeoff_simple_on"  "simple_overlap" "true"  >/dev/null
make_cfg_py "$CONFIG" "$c2" "tradeoff_simple_off" "simple_overlap" "false" >/dev/null
make_cfg_py "$CONFIG" "$c3" "tradeoff_jaccard_on" "jaccard"        "true"  >/dev/null
make_cfg_py "$CONFIG" "$c4" "tradeoff_jaccard_off" "jaccard"       "false" >/dev/null

# Run them all
run_one "$c1"
run_one "$c2"
run_one "$c3"
run_one "$c4"

# Summarize
summarize