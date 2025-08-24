# CoCoPipe (cocop)

**Cross‑Modal Coherence in Multi‑Agent Vision‑and‑Text Pipelines**

> Paper‑aligned module for: **“CoCoPipe: Cross‑Modal Coherence in Multi‑Agent Vision‑and‑Text LLM Pipelines.”**

---

## 💡 What is CoCoPipe?

CoCoPipe studies **how well multiple agents across text and vision modalities stay consistent** when working together in complex pipelines. It provides **metrics, benchmarks, and adapters** to measure and improve *coherence* between OCR outputs, reasoning steps, synthesis agents, and final answers.

Where `xpipe` focuses on explainability and `metap` on adaptive routing, `cocop` zeroes in on **cross‑modal consistency** — making sure that what the vision model reads is what the text model reasons over and what the synthesis model outputs.

---

## ✨ Core Capabilities

* **Cross‑Modal Task Partitioning** — define standardized roles across modalities (e.g., OCR, fusion, synthesizer, evaluator).
* **Coherence Metrics** — quantitative scores for semantic/structural alignment across agents (e.g., OCR vs. reasoning vs. synthesis).
* **Stress‑Testing** — evaluate pipelines on noisy, adversarial, or degraded visual inputs (handwriting, low‑res scans).
* **Feedback Loops** — optional mechanisms for agents to send *coherence feedback* to upstream peers.
* **Benchmarks & Datasets** — curated forms and documents with varying quality to test robustness.

---

## 🧱 Architecture

```
   Image/Form → [ OCR Agent ] → raw JSON/text → [ Fusion Agent ]
                                   │
                  question/context ─┼─────────→ [ Reasoning Agent ]
                                   │
                                   ▼
                               [ Synthesizer ]
                                   │
                                   ▼
                              [ Evaluator (Judge) ]
                                   │
                               Coherence Score(s)
```

**Packages**

* `cocop.ocr` — wrappers for vision models (OCR agents).
* `cocop.fusion` — combine raw OCR text with structured schemas.
* `cocop.reason` — reasoning/refinement agents.
* `cocop.synth` — synthesis agents.
* `cocop.eval` — coherence metrics + evaluation harness.
* `cocop.datasets` — noisy/clean benchmark sets.

---

## 🚀 Quickstart

### 1) Install

```bash
pip install -e ./cocop
```

### 2) Minimal Run

```python
from cocop import Pipeline, datasets

pipe = Pipeline(
  ocr_model="gemma-vision-27b",
  reason_model="qwen3-14b",
  synth_model="llama-3.3-70b",
  eval_model="deepseek-v3"
)

form = datasets.load("insurance_form_noisy")
result = pipe.run(image=form.image, question="What is the applicant’s email?")

print(result.answer)
print(result.metrics["coherence_score"])
```

---

## ⚙️ Configuration

```yaml
experiment: cocop_insurance_v1
pipeline: vision_text
ocr_model: gemma-vision-27b
reason_model: qwen3-14b
synth_model: llama-3.3-70b
judge_model: deepseek-v3
metrics:
  coherence: true
  faithfulness: true
  retrieval: false
stress_test:
  noise_levels: [clean, medium, heavy]
logging:
  logdir: output/cocop
```

---

## 📦 Data & Logging

```
output/
└── cocop/
    ├── runs/        # execution traces
    ├── metrics/     # coherence/relevance/faithfulness scores (CSV)
    ├── samples/     # per‑query annotated inputs/outputs
    └── evals/       # summary reports
```

---

## 📊 Coherence Metrics

* **Lexical Overlap**: do OCR terms survive to synthesis?
* **Entity Consistency**: are key entities (names, IDs, dates) consistent across steps?
* **Structural Alignment**: schema conformity (e.g., Pydantic contract).
* **Cross‑Agent Agreement**: evaluator checks if reasoning contradicts OCR.
* **Robustness**: drop in coherence under noise/degradation.

---

## 🔬 Evaluation

Run built‑in evaluation suites:

```bash
python scripts/run_cocop.py \
  --dataset insurance_forms --noise heavy \
  --out output/cocop/evals
```

Generates CSVs + plots with coherence breakdown per role + per noise level.

---

## 🧩 Adapters

* `cocop.adapters.rag` — test RAG coherence when OCR text is used as context.
* `cocop.adapters.multi_agent` — apply coherence metrics to multi‑agent reasoning traces.
* `cocop.adapters.vision_text` — default adapter for OCR → reasoning → synthesis.

---

## 🛡️ Constraints & Safety

* Noisy input handling (handwriting, low‑res, artifacts).
* Explicit flagging of uncertain/ambiguous fields for **human review**.
* Privacy: local logging; redact sensitive fields.

---

## 🗺️ Roadmap

* Learnable coherence metric (contrastive embeddings between OCR & synthesis).
* Active learning with *coherence‑based feedback*.
* Cross‑pipeline generalization (train coherence models on forms, apply to RAG).
* Plug‑in visualization dashboard (like xpipe) for side‑by‑side OCR vs. synthesis.

---

## 📚 Citation (paper preprint)

```
@inproceedings{mesabo2025cocopipe,
  title     = {CoCoPipe: Cross-Modal Coherence in Multi-Agent Vision-and-Text LLM Pipelines},
  author    = {Messou, Franck J. A. and Collaborators},
  booktitle = {Proc. of <Venue>},
  year      = {2025}
}
```

---

## 📝 License

This project uses the **MIT License** (see `LICENSE`).

---

## 🙌 Acknowledgements

Built on top of the Agentic RAG, Multi‑Agent, and Vision pipelines in this repo. Special thanks to early testers of noisy document OCR and cross‑modal evaluation.
