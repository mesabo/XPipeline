#!/usr/bin/env bash
# ----------------------------------------------------------------------
# serve_dashboard.sh
# Convenience wrapper to launch the X-Pipe Streamlit dashboard
#
# Usage:
#   bash xpipe/scripts/serve_dashboard.sh [PORT]
#
# Notes:
# - Default port is 8501; override by passing a number as first argument.
# - Requires `streamlit` installed in the current conda environment.
# ----------------------------------------------------------------------

set -euo pipefail

PORT="${1:-8501}"

# Run Streamlit app on chosen port, bind to 0.0.0.0 for remote access
streamlit run xpipe/scripts/serve_dashboard.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0