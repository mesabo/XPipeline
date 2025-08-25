# X-Pipe — An Explainable Evaluation Framework for Multi-Stage LLM Pipelines

Paper-aligned module for “X-Pipe: An Explainable Evaluation Framework for Multi-Stage LLM Pipelines.”

Badges
- Status: Draft / Research
- License: MIT

---

## Table of Contents
- [Overview](#%F0%9F%94%8E-overview)
- [Getting Started](#%E2%9C%85-getting-started)
  - [Environment](#environment)
  - [Quickstart](#quickstart)
  - [Sweeps and Scripts](#sweeps-and-scripts)
- [Architecture](#%F0%9F%A1%B1-architecture)
- [Directory Structure](#%F0%9F%93%A6-directory-structure)
- [Configuration](#%E2%9A%99%EF%B8%8F-configuration)
- [Outputs](#%F0%9F%94%AC-outputs)
- [Ablations](#%F0%9F%A7%AA-ablations)
  - [Trade-off Ablations](#trade-off-ablations)
- [Visualization & Dashboard](#%F0%9F%93%8A-visualization--dashboard)
- [Extended Configs & Workflows](#%E2%9D%99%EF%B8%8F-extended-configs--workflows)
- [Notebooks & Examples](#%F0%9F%A9%97-notebooks--examples)
- [Troubleshooting](#%F0%9F%9A%B0-troubleshooting)
- [Roadmap](#%F0%9F%97%BA-roadmap)
- [Reliability & Privacy](#%F0%9F%9B%A1%EF%B8%8F-reliability--privacy)
- [Citation, Acknowledgements & License](#%F0%9F%93%9A-citation-acknowledgements--license)

---

## 🔎 Overview
![Overview](/docs/overview.png "Overview")
![Overview](/docs/groups.png "Group comparison")
![Overview](/docs/tables.png "Tables")

X-Pipe instruments multi-stage LLM pipelines (RAG, multi-agent, vision-text, judges) to make them transparent, debuggable, and comparable. It captures per-stage decision rationales, logs traces and metrics, and supports causal attribution (ablations) to identify error sources.

Key features
- Causal attribution via ablations
- Thin adapters to integrate into existing workflows
- Tracing of spans, events, artifacts, and token usage
- Metric logging (relevance, faithfulness, latency, cost)
- Example notebooks for reproducible demos

Notebooks
- `01_agentic_RAG.ipynb` — Agentic RAG pipeline
- `02_multi_agent_system.ipynb` — Multi-agent research workflows
- `03_vision_reasoning.ipynb` — OCR + reasoning over scanned forms

---

## ✅ Getting Started

### Environment
Choose one Conda environment:

CPU-only
```bash
conda env create -f env_llms_cpu.yml
conda activate llms
```

GPU (CUDA 12.1 via Conda)
```bash
conda env create -f env_llms_gpu.yml
conda activate llms
```

Tip for offline servers
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/shared/hf_cache
```

### Quickstart

Single run (RAG demo)
```bash
python xpipe/main.py --config xpipe/configs/experiment_rag.yaml
```

Outputs (examples)
- `output/xpipe/runs/<stamp>_xpipe_rag_*.jsonl`  
- `output/xpipe/runs/<stamp>_xpipe_rag_*.pretty.json` (human readable)  
- `output/xpipe/metrics/xpipe_rag_*.csv`

### Sweeps and Scripts

Sweep multiple LLMs
```bash
# All synthesizer models
./xpipe/scripts/run_all_llms.sh

# Hugging Face models only
./xpipe/scripts/run_all_llms.sh synth "hf/"

# Sweep judges only
./xpipe/scripts/run_all_llms.sh judge

# Full grid (synth × judge, limit to 8 combos)
./xpipe/scripts/run_all_llms.sh grid "" "" 8
```

The script reports: `[sweep] Prepared N configs in output/xpipe/tmp_configs`.

---

## 🧱 Architecture

Pipeline (high level)
```
request → Stage 1 (Retriever / Router)
        → Stage 2 (Reasoner / Synthesizer)
        → Stage 3 (Judge / Safety)
```

Core modules
- `xpipe.trace` — spans, events, artifacts, token usage
- `xpipe.metrics` — metric registry, CSV export
- `xpipe.attribution` — ablation and responsibility mapping
- `xpipe.runners` — demo runners (RAG, multi-agent, vision-text)
- `xpipe.llm_backends` — HF + optional Ollama dispatch

Paper claims implemented
- Explainable tracing per stage (spans, artifacts, token counts)
- Holistic metrics (relevance, faithfulness, latency, cost)
- Causal attribution via ablations
- Flexible model selection per stage via config

---

## 📦 Directory Structure
```
xpipe/
├── main.py                    # Entry point (RAG demo + trace/metrics)
├── xpipe/                     # Core library package
│   ├── __init__.py
│   ├── trace.py               # Spans, events, JSONL writer
│   ├── metrics.py             # MetricLog (CSV output)
│   ├── attribution.py         # Ablation/grid utilities
│   ├── llm_backends.py        # HF + optional Ollama dispatch
│   └── runners/               # Adapters for demos
│       ├── rag.py
│       ├── multi_agent.py
│       └── vision_text.py
├── configs/
│   ├── experiment_rag.yaml    # Pipeline, corpus, queries, stage → model
│   └── models.yaml            # Model handles (HF, optional Ollama)
├── scripts/
│   ├── run_all_llms.sh        # Sweeps models, prepares configs
│   └── run_xpipe.sbatch       # Optional SLURM launcher
└── output/xpipe/
    ├── runs/                  # JSONL traces + pretty JSON
    ├── metrics/               # CSV metrics
    ├── ablations/             # Ablation grids
    └── figs/                  # Plots
```

---

## ⚙️ Configuration

Example `experiment_rag.yaml` (excerpt)
```yaml
name: xpipe_rag_free
pipeline: rag
logdir: output/xpipe
retriever:
  top_k: 3
llms:
  synthesize:
    model: hf/gpt2
    params:
      max_new_tokens: 200
      temperature: 0.2
  judge:
    model: heuristic
corpus:
  - {id: d1, text: "Paris is the capital of France."}
queries:
  - {id: q1, text: "What is the capital of France?"}
```

Example `models.yaml` (excerpt)
```yaml
hf/distilgpt2:
  backend: hf
  id: distilgpt2
  params: {max_new_tokens: 160, temperature: 0.0}
hf/Qwen2.5-0.5B-Instruct:
  backend: hf
  id: Qwen/Qwen2.5-0.5B-Instruct
  params: {max_new_tokens: 200, temperature: 0.2}
```

Notes
- Stages (e.g., `synthesize`, `judge`) can independently select models from `models.yaml`.
- Stage-specific params override defaults from `models.yaml`.
- `models_path` in `experiment_rag.yaml` can point to custom model files.

---

## 🔬 Outputs

Traces (`output/xpipe/runs/*.jsonl`, `*.pretty.json`) contain:
- Per-stage spans (start/log/end)
- Artifacts (prompts, contexts)
- Latency and token usage

Example trace schema
```json
{
  "experiment": "xpipe_rag_free",
  "run_id": "4766ed64",
  "run_tags": {"pipeline": "rag"},
  "started_ts_ms": 172449...,
  "wall_time_ms": 1543,
  "token_usage": {"prompt": 345, "completion": 214},
  "events": [...]
}
```

Metrics (`output/xpipe/metrics/*.csv`)
- Columns: pipeline, item (query ID), relevance, faithfulness, latency_ms, cost_usd
- Custom metrics can be added via `MetricLog.add()`

---

## 🧪 Ablations

Run ablations to analyze component contributions:
```python
from xpipe.attribution import ablate

def my_pipeline(item, retriever, judge):
    # Pipeline logic
    return {"scores": ...}

df = ablate(
    pipeline_fn=my_pipeline,
    dataset=["q1", "q2", "q3"],
    factors={"retriever": ["bm25", "dense", "hybrid"],
             "judge": ["heuristic", "hf/gpt2"]}
)
df.to_csv("output/xpipe/ablations/ablate.csv")
```

⸻

### 🔄 Trade-off Ablations

Besides Python-level ablate(), we provide ready Bash scripts for systematic trade-off evaluation.

Retriever × Judge trade-off

Compares two retrievers (simple_overlap, jaccard) × judge enabled/disabled.

```bash
bash xpipe/scripts/ablate_tradeoff.sh xpipe/configs/experiment_rag.yaml
```

It automatically:
- clones the base config four times with retriever/judge settings,
- runs all four configs,
- aggregates results from output/xpipe/metrics/*.csv,
- prints a summary table:

```
run	retriever	judge	mean_ground	mean_rougeL	mean_f1	mean_latency_ms	synth_tokens	judge_tokens
tradeoff_simple_on	simple_overlap	True	0.88	0.15	0.24	3650.2	951	976
tradeoff_simple_off	simple_overlap	False	0.85	0.14	0.23	3201.5	951	0
tradeoff_jaccard_on	jaccard	True	0.87	0.16	0.25	3741.3	951	976
tradeoff_jaccard_off	jaccard	False	0.84	0.13	0.21	3159.9	951	0
```

This answers “Would removing a judge or switching retrievers improve the trade-off?”

Outputs:
- configs are generated under /tmp/.../*.yaml (temporary),
- metrics always land in `output/xpipe/metrics/`.

---

## 📊 Visualization & Dashboard

### CLI Inspector

Minimal dependency-light CLI to inspect runs and plot summary figures.

```bash
python xpipe/scripts/cli_inspect.py --figs
```

Produces under `output/xpipe/figs/`:
- `mean_grounding_by_run.png`
- `latency_vs_grounding.png`
- `calibration_curve.png` + `calibration_bins.csv` (if confidence/correct are logged)

### Streamlit Dashboard

Interactive UI for browsing results.

```bash
streamlit run xpipe/scripts/serve_dashboard.py
```

Features:
- Multi-run comparison (select runs or “all runs”)
- Group comparison: by retriever, judge_enabled, or run
- Bar charts with mean ± stdev whiskers
- Latency vs. grounding scatter
- Token & cost overview
- Raw trace viewer

Access via:
- Local VSCode Remote SSH: http://localhost:8501
- Or forward ports manually: `ssh -L 8501:127.0.0.1:8501 user@server`

---

## ⚙️ Extended Configs

Example config with judge toggle + reference answers for metrics:

```yaml
name: xpipe_rag_free
pipeline: rag
logdir: output/xpipe

retriever:
  name: simple_overlap   # or jaccard
  top_k: 3

llms:
  synthesize:
    model: hf/gpt2
    params: {max_new_tokens: 160, temperature: 0.1}
  judge:
    enabled: true        # set to false to ablate the judge
    model: hf/distilgpt2
    params: {max_new_tokens: 8, temperature: 0.0}

prompt:
  budget_tokens: 900
  ctx_per_doc_tokens: 300

dataset:
  path: datasets/tbmp_2024/chunks.jsonl

queries:
  - id: q1
    text: "What is the role of a motion to compel in TTAB discovery?"
    ref:  "A motion to compel asks the Board to order a party to provide required discovery responses..."
  - id: q2
    text: "When are sanctions appropriate relative to motions to compel?"
    ref:  "Sanctions may be considered if a party fails to comply with discovery obligations..."

outputs:
  run_jsonl: ""
  metrics_csv: ""
```

This format supports:
- Judge ablation (`enabled: true/false`)
- Retriever swap (`simple_overlap` / `jaccard`)
- Reference answers (`ref:`) → enables ROUGE-L, token-F1, calibration.

---

## 🧩 Extended Example Workflow
1. Run baseline RAG:
```bash
python xpipe/main.py --config xpipe/configs/experiment_rag.yaml
```

2. Run all models in sweep:
```bash
bash xpipe/scripts/run_all_llms.sh
```

3. Run trade-off ablations:
```bash
bash xpipe/scripts/ablate_tradeoff.sh xpipe/configs/experiment_rag.yaml
```

4. Inspect results:
```bash
python xpipe/scripts/cli_inspect.py --figs
streamlit run xpipe/scripts/serve_dashboard.py
```

---

## 🧩 Notebooks & Examples
Reuse core components in notebooks:
```python
from xpipe.trace import Trace
from xpipe.metrics import MetricLog
from xpipe.llm_backends import call_llm
```

---

## 🧰 Troubleshooting

- Ollama connection refused: restrict to Hugging Face models:
  ```bash
  ./xpipe/scripts/run_all_llms.sh synth "hf/"
  ```
- GLIBCXX / pandas errors: use provided Conda envs (`env_llms_cpu.yml`, `env_llms_gpu.yml`).
- Out of VRAM: prefer `hf/distilgpt2` or `hf/gpt2` for lower memory usage.
- Offline HF weights: set `TRANSFORMERS_OFFLINE=1` and `HF_HOME=/path/to/shared/hf_cache`.

---

## 🗺️ Roadmap
- Model-agnostic calibration (temperature scaling, conformal control)
- Failure clustering from traces
- Cross-run significance tests / confidence intervals
- Integration with W&B, MLflow, Grafana

---

## 🛡️ Reliability & Privacy
- Local-only logging by default
- Optional PII redaction in traces
- Deterministic seeds for evaluators and ablations

---

## 📚 Citation
```
@inproceedings{mesabo2025xpipe,
  title     = {X-Pipe: An Explainable Evaluation Framework for Multi-Stage LLM Pipelines},
  author    = {Messou, Franck J. A. and Collaborators},
  booktitle = {Proc. of <Conference to be continued>},
  year      = {2025}
}
```

---

## 🙌 Acknowledgements
Built on three exemplar pipelines. Thanks to contributors and reviewers for guidance on instrumentation and metrics.

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.