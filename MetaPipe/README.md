# MetaPipe (metap)

**Adaptive Meta‑Orchestration for Task‑Aware LLM Pipeline Routing**

> Paper‑aligned module for: **“MetaPipe: Adaptive Meta‑Orchestration for Task‑Aware LLM Pipeline Routing.”**

---

## 💡 What is MetaPipe?

MetaPipe learns **which model to use for which pipeline role** (retrieval, synthesis, judging, OCR, fusion, etc.) **given the task context and constraints** (quality, latency, cost). Instead of static assignments, MetaPipe uses a **meta‑controller** that observes features of the task and **routes** each stage to the best LLM/tool. It can **reconfigure mid‑run** when confidence drops or inputs degrade.

Works across your three exemplar pipelines:

* Agentic RAG (deep document analysis)
* Multi‑Agent Scientific Research
* Vision + Text (scanned/handwritten forms)

---

## ✨ Core Capabilities

* **Task & Context Embeddings** — encode signals like query/domain, length, OCR quality, retrieval entropy, past success/latency/cost.
* **Meta‑Controller (Policy)** — learns a routing policy over a *candidate set* of LLMs/tools per role (e.g., Fast‑LLM vs. Strong‑LLM).
* **Dynamic Reconfiguration** — mid‑pipeline switching based on uncertainty or budget remaining ("escalate on fail").
* **Multi‑Objective Optimization** — trade off quality, latency, and dollars; supports hard constraints (e.g., max 1.5s).
* **Plug‑and‑Play** — wrap existing pipelines with thin adapters; no heavy refactors.

---

## 🧱 Architecture

```
              ┌───────────── Candidates per role ─────────────┐
Query/Input →  Role 1: {LLM_A, LLM_B, Tool_X}  ←─┐
               Role 2: {LLM_C, LLM_D}           ─┼─>  Meta‑Controller π( a | ϕ(task, state) )
               Role 3: {Judge1, Judge2, None}   ←┘
                                 │ actions (model picks)
                                 ▼
                         Executed Pipeline
                                 │ feedback
                                 ▼
                           Experience Buffer → Policy Update
```

**Packages**

* `metap.features` — builders for task/context embeddings (static + online signals).
* `metap.policy` — meta‑controller implementations: heuristic, contextual bandit, RL, or supervised matching network.
* `metap.runtime` — router + executors + escalation logic; budget/SLAs.
* `metap.evals` — multi‑objective metrics, regret curves, Pareto plots, success\@budget.
* `metap.adapters` — wrappers for RAG, multi‑agent, and vision‑text pipelines.

---

## 🚀 Quickstart

### 1) Install

```bash
pip install -e ./metap
```

### 2) Define candidates per role

```python
CANDIDATES = {
  "retrieve": [bm25, hybrid, dense],
  "synthesize": [fast_llm, strong_llm],
  "judge": [none, gpt_judge, two_stage_critique],
}
```

### 3) Build features & policy

```python
from metap.features import build_features
from metap.policy import ContextualBandit
from metap.runtime import Router

policy = ContextualBandit(candidates=CANDIDATES)
router = Router(policy=policy)

ctx = build_features(query=q, retrieved_k=10, domain="legal", budget_usd=0.01)
plan = router.plan(ctx)         # e.g., {retrieve: hybrid, synthesize: fast_llm, judge: gpt_judge}
result = router.execute(plan, query=q)
```

### 4) Online learning (bandit/RL)

```python
reward = result.metrics["quality"] - 0.1*result.metrics["cost_usd"]
policy.update(context=ctx, action=plan, reward=reward)
```

---

## ⚙️ Configuration

YAML example:

```yaml
experiment: metap_rag_v1
pipeline: rag
budget_usd: 0.02
sla_ms: 1800
candidates:
  retrieve: [bm25, dense, hybrid]
  synthesize: [fast_llm, strong_llm]
  judge: [none, gpt_judge]
policy:
  type: contextual_bandit   # {heuristic, contextual_bandit, rl, matcher}
  features: [len_query, domain_id, retriever_entropy, ocr_quality, hist_success, hist_cost]
reconfig:
  escalate_on_low_conf: true
  escalate_threshold: 0.35
logging:
  logdir: output/metap
```

---

## 📦 Data & Logging

```
output/
└── metap/
    ├── plans/      # chosen actions per role
    ├── runs/       # execution traces (compatible with xpipe)
    ├── policy/     # snapshots of learned policy weights
    └── evals/      # CSVs: quality, latency, cost, regret
```

MetaPipe is **xpipe‑compatible**: you can record `Trace`/`MetricLog` during execution for unified dashboards.

---

## 📊 Evaluation

* **Quality**: task success / correctness / VQA‑style scoring.
* **Latency & Cost**: per‑stage + end‑to‑end; budget adherence.
* **Regret**: vs. best static policy and oracle per‑role picks.
* **Pareto**: quality–cost and quality–latency frontiers.
* **Escalation Efficacy**: success lift vs. extra spend.

Reproduce baseline vs. MetaPipe:

```bash
python scripts/run_metap.py \
  --pipeline rag --dataset docs-mini \
  --policy contextual_bandit \
  --out output/metap
```

---

## 🔬 Policies Included

* **Heuristic**: hand‑crafted if/else on features (fast baseline).
* **Contextual Bandit**: LinUCB/Thompson sampling with engineered features.
* **RL Controller**: (optional) actor‑critic over multi‑step routing with budget.
* **Matcher**: supervised model from offline logs mapping contexts → best actions.

You can start with **Heuristic** or **Bandit** (minimal code), then graduate to **RL**.

---

## 🧩 Adapters

* `metap.adapters.rag` — wraps retriever/synthesizer/judge.
* `metap.adapters.multi_agent` — routes agent roles and escalation.
* `metap.adapters.vision_text` — gates OCR model choice and fusion.

Adapters expose a common interface so policies stay pipeline‑agnostic.

---

## 🛡️ Constraints & Safety

* **Budgets**: hard and soft caps (drop/abstain when exceeded).
* **Confidence‑aware Routing**: backoff/abstain, or escalate.
* **Privacy**: redact traces; local logs by default.

---

## 🗺️ Roadmap

* Learned feature encoders from raw traces (sequence models over spans)
* Multi‑objective RL with constraint handling (Lagrangian / CPO)
* Cross‑pipeline transfer (train on RAG, reuse on vision‑text)
* Policy distillation into tiny heuristics for edge deployment

---

## 📚 Citation (paper preprint)

```
@inproceedings{mesabo2025metapipe,
  title     = {MetaPipe: Adaptive Meta-Orchestration for Task-Aware LLM Pipeline Routing},
  author    = {Messou, Franck J. A. and Collaborators},
  booktitle = {Proc. of <Venue>},
  year      = {2025}
}
```

---

## 📝 License

Follows repository license (**BSD‑3‑Clause**).

---

## 🙌 Acknowledgements

Built on top of the project’s RAG, multi‑agent, and vision‑text pipelines. Thanks to contributors for early discussions on routing, bandits, and escalation policies.
