# X-Pipe (xpipe)

**An Explainable Evaluation Framework for Multi‑Stage LLM Pipelines**

> Paper-aligned module for the project: **“X‑Pipe: An Explainable Evaluation Framework for Multi‑Stage LLM Pipelines.”**

---

## 🔎 What is X‑Pipe?

X‑Pipe makes complex LLM pipelines **transparent, debuggable, and comparable**. It instruments every stage (RAG, multi‑agent reasoning, vision‑text OCR/fusion, safety checks, judges, etc.), logs *why* decisions were made, and visualizes *where* errors originate. It also supports **causal attribution** and **ablations** to quantify each component’s contribution to the final outcome.

X‑Pipe plugs into your existing notebooks/modules:

* `01_agentic_RAG.ipynb` (agentic RAG)
* `02_multi_agent_system.ipynb` (research multi‑agent)
* `03_vision_reasoning.ipynb` (vision + reasoning for scanned forms)

---

## ✨ Core Capabilities

* **Agentic Trace Logging** — capture prompts, intermediate outputs, tool/RAG calls, costs, latencies, and confidence scores per stage.
* **Causal Attribution** — attribute success/failure to specific components (router, retriever, synthesizer, judge, OCR, fusion) using perturbation- and proxy‑based methods.
* **Ablation & What‑If Analysis** — disable/replace agents and quantify performance deltas ("map of responsibility").
* **Unified Metrics** — relevance, faithfulness, answer correctness, latency, cost, and calibration (selective risk / abstention).
* **Interactive Dashboard** — inspect runs, compare pipelines, trace error propagation.

---

## 🧱 Architecture

```
request ─▶ Stage 1 (Retriever/Router) ─▶ Stage 2 (Reasoner/Synthesizer) ─▶ Stage 3 (Judge/Safety)
          │                              │                                  │
          └─▶ xpipe.trace ───────────────┴────────▶ xpipe.metrics ───────────┴─▶ xpipe.attribution
                                               └──────────────────────────────▶ xpipe.dashboard
```

**Packages**

* `xpipe.trace` — structured events, spans, artifacts (prompts, contexts, images, tool calls).
* `xpipe.metrics` — task metrics (accuracy, BLEU/ROUGE, faithfulness, calibration), runtime/cost.
* `xpipe.attribution` — component‑level responsibility estimates (perturbation, swap‑outs, SHAP‑style proxies where applicable).
* `xpipe.dashboard` — local UI (Streamlit/Gradio) for filtering runs, drilling into stages, and comparing variants.
* `xpipe.runners` — thin adapters for your existing pipelines (RAG, multi‑agent, vision‑text).

---

## 🚀 Quickstart

### 1) Install

```bash
# from the repo root
pip install -e .
# or, module-only
pip install -e ./xpipe
```

### 2) Instrument your pipeline

```python
from xpipe.trace import Trace
from xpipe.metrics import MetricLog

trace = Trace(experiment="rag_v1", run_tags={"dataset":"docs-mini"})
metrics = MetricLog()

with trace.span("retrieve") as sp:
    ctx = retriever(query)
    sp.log({"k": len(ctx), "latency_ms": 12})

with trace.span("synthesize") as sp:
    answer = llm_synth(query, ctx)
    sp.log({"prompt_tokens": 356, "completion_tokens": 221})

metrics.add(
    relevance=score_relevance(answer, ctx),
    faithfulness=score_faithfulness(answer, ctx),
    latency_ms=trace.wall_time_ms,
    cost_usd=estimate_cost(trace.token_usage),
)

trace.save(); metrics.save()
```

### 3) Run ablations

```python
from xpipe.attribution import ablate

results = ablate(
    pipeline_fn=my_pipeline,              # callable(query) -> answer
    factors={
        "retriever": ["bm25", "hybrid", "dense"],
        "judge": ["none", "gpt-judge", "critique-2stage"],
    },
    dataset=queries[:100]
)
results.to_csv("./output/xpipe_ablations.csv")
```

### 4) Launch the dashboard

```bash
python -m xpipe.dashboard --logdir ./output/xpipe_logs
```

---

## 📦 Data & Logging Layout

```
output/
└── xpipe/
    ├── runs/
    │   └── 2025-08-24_13-05-22_rag_v1.jsonl   # unified trace (events/spans/artifacts)
    ├── metrics/
    │   └── rag_v1_metrics.csv                 # per-run metrics (task + system)
    ├── ablations/
    │   └── rag_v1_ablate.csv                  # grid/what-if outcomes
    └── figs/
        ├── trace_graph.png
        └── calibration_plot.png
```

---

## 📊 Metrics (Built‑ins)

* **Task Quality**: exact/partial match, ROUGE‑L, BLEU, answerable F1, VQA‑style scores.
* **Groundedness**: faithfulness (evidence overlap), attribution of citations, hallucination rate.
* **System**: total latency, per‑stage latency, token counts, \$\$ cost, GPU/CPU utilization (optional).
* **Calibration**: confidence vs. accuracy (ECE), selective prediction (risk‑coverage).

> Plug your own scorers via the `xpipe.metrics.Registry`.

---

## 🧪 Experiments

* **Pipelines**: agentic‑RAG, scientific multi‑agent, vision‑text OCR/fusion.
* **Comparisons**: fixed vs. adaptive routing, single‑judge vs. multi‑judge, single vs. hybrid retrieval.
* **Stress Tests**: noisy OCR, ambiguous instructions, long‑context documents, cross‑domain generalization.

Reproduce with:

```bash
python scripts/run_experiment.py \
  --pipeline rag \
  --dataset docs-mini \
  --variants retriever=hybrid judge=gpt-judge \
  --out output/xpipe
```

---

## ⚙️ Configuration

YAML/CLI for reproducibility. Example:

```yaml
experiment: rag_v1
logdir: output/xpipe
pipeline: rag
variants:
  retriever: [bm25, dense, hybrid]
  judge: [none, gpt-judge]
metrics:
  - relevance
  - faithfulness
  - latency_ms
  - cost_usd
```

---

## 🖥️ Dashboard Features

* Run table with filters (pipeline, dataset, date, variants)
* Span graph (timeline per stage, with drill‑down)
* Error propagation view (stage → stage)
* Calibration & selective prediction plots
* Compare runs side‑by‑side (diff tokens/cost/latency/quality)

---

## 🧩 Adapters (Runners)

* `xpipe.runners.rag` — instrument retrieval, reranking, synthesis, judging.
* `xpipe.runners.multi_agent` — capture agent messages, roles, tool calls.
* `xpipe.runners.vision_text` — log OCR outputs, layout parsing, fusion steps.

Each runner exposes a minimal interface to wrap an existing pipeline without refactoring core logic.

---

## 🛡️ Reliability & Privacy

* Optional redaction of PII in traces.
* Local‑only logging by default; remote storage requires explicit opt‑in.
* Deterministic seeds for evaluators and ablations.

---

## 🗺️ Roadmap

* Model‑agnostic calibration (temperature scaling + conformal risk control)
* Automated failure clustering (trace → failure taxonomy)
* Cross‑run statistical tests and confidence intervals
* Export to W\&B / MLflow; Grafana datasource for system metrics

---

## 📚 Citation (paper preprint)

```
@inproceedings{mesabo2025xpipe,
  title     = {X-Pipe: An Explainable Evaluation Framework for Multi-Stage LLM Pipelines},
  author    = {Messou, Franck J. A. and Collaborators},
  booktitle = {Proc. of <Venue>},
  year      = {2025}
}
```

---

## 📝 License

This module follows the repository license (**BSD‑3‑Clause**).

---

## 🙌 Acknowledgements

Built on top of the project’s three exemplar pipelines (Agentic RAG, Multi‑Agent Research, Vision‑Text Reasoning). Thanks to contributors and reviewers for feedback on instrumentation, metrics, and dashboard design.
