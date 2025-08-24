# X-Pipe: Scripts — How to run each helper

This document describes the purpose and usage for each script in xpipe/scripts. Run commands from the repository root (project folder containing `xpipe/`).

Prerequisites
- Activate the provided conda environment:
```bash
# one-time (or inside run_all_llms.sh which sources this)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llms
```
- Ensure `xpipe/main.py` exists and runs (it is the pipeline entrypoint).
- Common paths:
  - Experiment config: `xpipe/configs/experiment_rag.yaml`
  - Models registry: `xpipe/configs/models.yaml`
  - Metrics: `output/xpipe/metrics/`
  - Runs/traces: `output/xpipe/runs/`
  - Figures: `output/xpipe/figs/`
  - Temp configs (sweeps): `output/xpipe/tmp_configs/`

Quick recommended workflow
1. (Optional) Build demo dataset:
   - `python xpipe/scripts/build_tbmp_dataset.py`
2. Run a baseline RAG:
   - `python xpipe/main.py --config xpipe/configs/experiment_rag.yaml`
3. Sweep models / ablate / inspect / visualize:
   - Use `run_all_llms.sh`, `ablate_tradeoff.sh`, `ablate_retriever_topk.py`, `cli_inspect.py`, `plot_metrics.py`, or the Streamlit dashboard.

Scripts index & usage

- run_all_llms.sh
  - Wrapper that activates the `llms` conda env and calls `sweep_llms.py`.
  - Usage:
    ```bash
    # default: synthesizer sweep
    ./xpipe/scripts/run_all_llms.sh
    # only HF synth handles
    ./xpipe/scripts/run_all_llms.sh synth "hf/"
    # only judge stage sweep
    ./xpipe/scripts/run_all_llms.sh judge
    # grid sweep but limit to first 8 combos
    ./xpipe/scripts/run_all_llms.sh grid "" "" 8
    ```
  - Internals: sets MODE, INCLUDE, EXCLUDE, LIMIT then runs `python -u xpipe/scripts/sweep_llms.py`.

- sweep_llms.py
  - Generates temp configs for many model combinations and runs `xpipe/main.py` on each.
  - Main options:
    - --mode {synth,judge,grid} (default synth)
    - --include / --exclude (substring filters)
    - --limit N (cap number of runs)
    - --backends (comma list, default "hf,ollama")
    - --dry-run (only print planned configs)
  - Output: writes configs under `output/xpipe/tmp_configs/` and invokes `xpipe/main.py` for each.
  - Example:
    ```bash
    python xpipe/scripts/sweep_llms.py --mode synth --include "hf/" --limit 10
    ```

- ablate_tradeoff.sh
  - Produces and runs four configs comparing retriever ∈ {simple_overlap, jaccard} × judge ∈ {on, off}.
  - Usage:
    ```bash
    bash xpipe/scripts/ablate_tradeoff.sh [CONFIG_PATH]
    # default CONFIG_PATH = xpipe/configs/experiment_rag.yaml
    ```
  - Notes:
    - Writes temporary YAMLs to a temp dir and runs `xpipe/main.py`.
    - Summarizes results by scanning `output/xpipe/metrics/*.csv`.
    - Ensure `PY` env var points to desired python if not `python`.

- ablate_retriever_topk.py
  - Grid ablation over `retriever.top_k`. For each k writes a temp config and runs `xpipe/main.py`.
  - Usage:
    ```bash
    python xpipe/scripts/ablate_retriever_topk.py \
      --exp xpipe/configs/experiment_rag.yaml \
      --models xpipe/configs/models.yaml \
      --main xpipe/main.py \
      --out output/xpipe/ablations/topk.csv \
      --ks 1 2 3 4 5
    ```
  - Notes:
    - `--main` should point to pipeline entry (usually `xpipe/main.py`).
    - Writes a CSV summary (`--out`) with mean grounding (mean relevance) per k.
    - Uses `models_path` injection so `main.py` can locate the registry.

- plot_metrics.py
  - Reads `output/xpipe/metrics/*.csv` and emits:
    - `mean_grounding_by_run.png` (bar of mean relevance per run)
    - `latency_vs_grounding.png` (scatter mean latency vs mean relevance)
  - Usage:
    ```bash
    python xpipe/scripts/plot_metrics.py --metrics_dir output/xpipe/metrics --out_dir output/xpipe/figs
    ```

- cli_inspect.py
  - Lightweight CLI inspector for metrics and recent run traces.
  - Features:
    - Per-run summary table (mean grounding/rouge/f1/latency/cost)
    - Retriever × Judge trade-off slice
    - Optional calibration/ECE if `confidence` + `correct` or `label` & `pred` present
    - Optional figures if `--figs` is passed (saved under `output/xpipe/figs/`)
  - Usage:
    ```bash
    python xpipe/scripts/cli_inspect.py
    python xpipe/scripts/cli_inspect.py --figs
    ```

- serve_dashboard.sh / serve_dashboard.py
  - `serve_dashboard.sh` is a convenience wrapper to run the Streamlit app on a chosen port.
  - `serve_dashboard.py` is the Streamlit dashboard:
    - Picks up metrics from `output/xpipe/metrics/*`
    - Provides group comparisons, bar+stdev charts, latency-quality scatter, per-run summary, and raw-table view.
    - Dependencies: streamlit, pandas, altair
  - Usage:
    ```bash
    # wrapper (defaults to 8501)
    bash xpipe/scripts/serve_dashboard.sh 8501
    # or run streamlit directly
    streamlit run xpipe/scripts/serve_dashboard.py --server.port 8501 --server.address 0.0.0.0
    ```
  - Access:
    - Local: http://localhost:8501
    - Remote: ssh -L 8501:127.0.0.1:8501 user@server

- build_tbmp_dataset.py
  - Download small TBMP (USPTO) PDF, extract text and write a tiny JSONL corpus to `datasets/tbmp_2024/chunks.jsonl`.
  - Usage:
    ```bash
    python xpipe/scripts/build_tbmp_dataset.py
    ```

Notes, tips & troubleshooting
- If a script calls `xpipe/main.py` and you want a different python interpreter, set `PYTHON_BIN` or pass `--python` (where supported).
- If outputs are missing after a run:
  - Check `output/xpipe/metrics/` for per-run CSVs named `<run_name>.csv`.
  - Check `output/xpipe/runs/` for JSONL trace files.
  - Confirm that `xpipe/main.py` honored `cfg["name"]` and `cfg["logdir"]` in temp configs.
- For offline servers, set:
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/hf_cache
```
- To limit memory usage prefer small HF models in `xpipe/configs/models.yaml` (e.g., `hf/distilgpt2`).

Where files land (summary)
- Temp configs for sweeps: output/xpipe/tmp_configs/
- Per-run metrics CSVs: output/xpipe/metrics/<run>.csv
- Per-run traces JSONL: output/xpipe/runs/<stamp>_<run>.jsonl
- Figures: output/xpipe/figs/
- Ablation CSVs (scripts may write): output/xpipe/ablations/

If you want, I can:
- Add example outputs (example CSV rows) for each script.
- Add a small "check-before-run" shell snippet to verify prerequisites.
- Create a top-level Makefile target to run common flows.