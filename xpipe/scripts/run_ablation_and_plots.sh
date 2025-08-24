#!/usr/bin/env bash
# ======================================================================
# run_ablation_and_plots.sh
# ----------------------------------------------------------------------
# Runs:
#   1) Ablation over retriever.top_k   -> output/xpipe/ablations/topk.csv
#   2) Plot metrics across runs        -> output/xpipe/figs/*.png
#
# Usage:
#   bash xpipe/scripts/run_ablation_and_plots.sh
#   # (no arguments needed — config path is fixed inside)
# ======================================================================

set -euo pipefail

# --- fixed settings ---
CONFIG_PATH="xpipe/configs/experiment_rag.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

ABLATE_SCRIPT="xpipe/scripts/ablate_retriever_topk.py"
PLOTS_SCRIPT="xpipe/scripts/plot_metrics.py"

LOGDIR="output/xpipe"
ABL_DIR="$LOGDIR/ablations"
FIG_DIR="$LOGDIR/figs"

# --- helpers ---
die () { echo "[ERROR] $*" >&2; exit 1; }
note() { echo -e "\n[INFO] $*"; }

# --- sanity checks ---
[ -x "$(command -v "$PYTHON_BIN")" ] || die "Python not found: $PYTHON_BIN"
[ -f "$CONFIG_PATH" ] || die "Config not found: $CONFIG_PATH"
[ -f "$ABLATE_SCRIPT" ] || die "Missing script: $ABLATE_SCRIPT"
[ -f "$PLOTS_SCRIPT" ] || die "Missing script: $PLOTS_SCRIPT"

mkdir -p "$ABL_DIR" "$FIG_DIR"

# --- 1) Ablation ---
note "Running ablation..."
"$PYTHON_BIN" "$ABLATE_SCRIPT" "$CONFIG_PATH"

TOPK_CSV="$ABL_DIR/topk.csv"
[ -f "$TOPK_CSV" ] && note "Ablation written to $TOPK_CSV" || die "Missing $TOPK_CSV"

# --- 2) Plots ---
note "Generating plots..."
"$PYTHON_BIN" "$PLOTS_SCRIPT"

PNG1="$FIG_DIR/mean_grounding_by_run.png"
PNG2="$FIG_DIR/latency_vs_grounding.png"

[ -f "$PNG1" ] && note "Plot available: $PNG1" || die "Missing $PNG1"
[ -f "$PNG2" ] && note "Plot available: $PNG2" || die "Missing $PNG2"

echo -e "\n[OK] All tasks completed."