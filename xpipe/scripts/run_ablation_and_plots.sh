#!/usr/bin/env bash
# run_ablation_and_plots.sh
# -------------------------------------------------------------------
# Runs: (1) retriever top-k ablation, (2) plotting over produced metrics.
# Robust to being called from anywhere. Mirrors sweep's path handling.
# -------------------------------------------------------------------
set -euo pipefail

# Resolve repo root from this script's location
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
XPIPE_DIR="$(dirname "$SCRIPT_DIR")"                    # .../xpipe
REPO_ROOT="$(dirname "$XPIPE_DIR")"                    # repo root

EXP_CFG="${XPIPE_DIR}/configs/experiment_rag.yaml"     # canonical experiment
MODELS_YAML="${XPIPE_DIR}/configs/models.yaml"         # canonical registry
MAIN_PY="${XPIPE_DIR}/main.py"                         # >>> correct entrypoint <<<
ABLATE_PY="${SCRIPT_DIR}/ablate_retriever_topk.py"     # companion script
PLOT_PY="${SCRIPT_DIR}/plot_metrics.py"                # companion plots

LOGROOT="${REPO_ROOT}/output/xpipe"
ABLAT_DIR="${LOGROOT}/ablations"
FIG_DIR="${LOGROOT}/figs"
MET_DIR="${LOGROOT}/metrics"

PY="${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PY="${PY:-python}"

echo
echo "[INFO] Using repo root: ${REPO_ROOT}"
echo "[INFO] Using models registry: ${MODELS_YAML}"
echo "[INFO] Using experiment cfg: ${EXP_CFG}"
echo "[INFO] Using main entrypoint: ${MAIN_PY}"
echo

# Sanity checks
[[ -f "$MAIN_PY" ]]    || { echo "[ERROR] Not found: $MAIN_PY"; exit 2; }
[[ -f "$EXP_CFG" ]]    || { echo "[ERROR] Not found: $EXP_CFG"; exit 2; }
[[ -f "$MODELS_YAML" ]]|| { echo "[ERROR] Not found: $MODELS_YAML"; exit 2; }
[[ -f "$ABLATE_PY" ]]  || { echo "[ERROR] Not found: $ABLATE_PY"; exit 2; }
[[ -f "$PLOT_PY" ]]    || { echo "[ERROR] Not found: $PLOT_PY"; exit 2; }

mkdir -p "$ABLAT_DIR" "$FIG_DIR" "$MET_DIR"

echo "[INFO] Running ablation with config: $EXP_CFG"
echo

# Run the ablation driver (writes CSV to output/xpipe/ablations/topk.csv)
"$PY" "$ABLATE_PY" \
  --exp "$EXP_CFG" \
  --models "$MODELS_YAML" \
  --main "$MAIN_PY" \
  --out "$ABLAT_DIR/topk.csv"

echo
echo "[INFO] Ablation completed → $ABLAT_DIR/topk.csv"
echo "[INFO] Generating plots from $MET_DIR/*.csv …"
echo

"$PY" "$PLOT_PY" \
  --metrics_dir "$MET_DIR" \
  --out_dir "$FIG_DIR"

echo
echo "[OK] Plots saved:"
echo " - $FIG_DIR/mean_grounding_by_run.png"
echo " - $FIG_DIR/latency_vs_grounding.png"