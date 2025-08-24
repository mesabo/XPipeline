#!/usr/bin/env bash
# Run all model handles from configs/models.yaml across one or more stages.

set -euo pipefail

# --- Always activate this env ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llms

# Defaults: sweep synthesizer only; no filters; no limit.
MODE="${1:-synth}"       # synth | judge | grid
INCLUDE="${2:-}"         # e.g. "hf/" or "ollama/"
EXCLUDE="${3:-}"         # e.g. "hf/"
LIMIT="${4:-0}"          # 0 = unlimited

ARGS=( --mode "$MODE" )
[[ -n "$INCLUDE" ]] && ARGS+=( --include $INCLUDE )
[[ -n "$EXCLUDE" ]] && ARGS+=( --exclude $EXCLUDE )
[[ "$LIMIT" != "0" ]] && ARGS+=( --limit "$LIMIT" )

python -u xpipe/scripts/sweep_llms.py "${ARGS[@]}"