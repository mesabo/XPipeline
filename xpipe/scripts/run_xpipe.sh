#!/usr/bin/env bash
# Always run the default RAG experiment for X-Pipe

set -euo pipefail

# --- Activate conda environment (always 'llms') ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llms

# Default config (change here if you want a different one)
CONFIG_FILE="xpipe/configs/experiment_rag.yaml"

echo "[X-Pipe] Using default config: $CONFIG_FILE"
python -u xpipe/main.py --config "$CONFIG_FILE"