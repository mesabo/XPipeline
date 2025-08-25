Got it 👍 — here’s a single top-level README.md that unifies all three papers/projects (XPipe, MetaPipe, CoCoPipe) in one place. Each section is paper-aligned, conference-ready, and self-contained.



README.md

# 🔬 LLM Pipeline Frameworks: XPipe, MetaPipe, and CoCoPipe

This repository hosts three research-driven frameworks that extend the design, evaluation, and orchestration of **multi-stage LLM pipelines**.  
Each module is aligned with a distinct conference paper:

- **XPipe** — *An Explainable Evaluation Framework for Multi-Stage LLM Pipelines*  
- **MetaPipe** — *Adaptive Routing for Task-Aware LLM Pipelines*  
- **CoCoPipe** — *Cross-Modal Coherence in Vision-and-Text LLM Pipelines*

All frameworks emphasize **public datasets**, **free/local models (HF, Ollama)**, and **reproducibility**.

---

## 📄 1. XPipe
![Overview](/docs/overview.png "Overview")
![Overview](/docs/groups.png "Group comparison")
![Overview](/docs/tables.png "Tables")

### Overview
XPipe instruments multi-stage LLM pipelines (RAG, multi-agent, vision-text) with **transparent tracing and metrics**.  
It supports **causal attribution**, ablations, and stage-wise explainability.

Dashboard available via  [https://xpipeserver.streamlit.app](https://xpipeserver.streamlit.app).

**Key features**
- Tracing spans, artifacts, token usage
- Metrics: relevance, faithfulness, latency, cost
- Causal attribution via ablations
- Pretty JSON logs for inspection

### Run
```bash
python xpipe/main.py --config xpipe/configs/experiment_rag.yaml

Outputs
	•	Runs: output/xpipe/runs/*.jsonl and *.pretty.json
	•	Metrics: output/xpipe/metrics/*.csv

Ablation Example

from xpipe.attribution import ablate
df = ablate(pipeline_fn=my_pipeline,
            dataset=["q1","q2"],
            factors={"retriever":["bm25","dense"],
                     "judge":["heuristic","hf/gpt2"]})

Citation

@inproceedings{mesabo2025xpipe,
  title={X-Pipe: An Explainable Evaluation Framework for Multi-Stage LLM Pipelines},
  author={Messou, Franck J. A. and Collaborators},
  year={2025}
}




📄 2. MetaPipe

Overview

MetaPipe introduces a meta-learning router that dynamically assigns LLMs to pipeline stages based on task context.
Instead of static configs, pipelines adapt in real time.

Key ideas
	•	Task embeddings for model selection
	•	Meta-learner for adaptive routing
	•	Dynamic pipeline reconfiguration
	•	Evaluation across quality, latency, and cost

Run

python metapipe/main.py --config metapipe/configs/experiment_meta.yaml

Outputs
	•	Runs: output/metapipe/runs/
	•	Metrics: output/metapipe/metrics/

Ablation Example

df = ablate(pipeline_fn=router_pipeline,
            dataset=["simple","hard"],
            factors={"router":["static","meta","oracle"]})

Citation

@inproceedings{mesabo2025metapipe,
  title={MetaPipe: Adaptive Routing for Task-Aware LLM Pipelines},
  author={Messou, Franck J. A. and Collaborators},
  year={2025}
}




📄 3. CoCoPipe

Overview

CoCoPipe extends orchestration to vision-and-text reasoning pipelines, ensuring semantic coherence across OCR, reasoning, and synthesis.

Key features
	•	Multi-modal task partitioning (OCR, fusion, synthesis)
	•	Coherence scoring across modalities
	•	Evaluation under adversarial/noisy inputs
	•	Co-training vision and language agents

Run

python cocopipe/main.py --config cocopipe/configs/experiment_vt.yaml

Outputs
	•	Runs: output/cocopipe/runs/
	•	Metrics: output/cocopipe/metrics/

Ablation Example

df = ablate(pipeline_fn=vision_text_pipeline,
            dataset=["clean","noisy"],
            factors={"ocr":["tesseract","hf/ocr-small"],
                     "reasoner":["hf/gpt2","hf/qwen2"]})

Citation

@inproceedings{mesabo2025cocopipe,
  title={CoCoPipe: Cross-Modal Coherence in Vision-and-Text LLM Pipelines},
  author={Messou, Franck J. A. and Collaborators},
  year={2025}
}




📦 Public Datasets
	•	TBMP (USPTO) — legal manual for RAG pipelines
	•	PubChem / ChEMBL — scientific protocols for multi-agent pipelines
	•	FUNSD / SROIE — scanned forms for OCR + reasoning

Configs point to datasets/ where preprocessed corpora (JSONL/CSV) are stored.



🛠️ Free / Local Models

All experiments are designed to run with free Hugging Face models (e.g., distilgpt2, gpt2, Qwen2.5-0.5B-Instruct) and optionally Ollama local backends (llama3.2:3b, qwen2.5:3b, deepseek-r1:1.5b).



📚 License

All code is under MIT LICENSE.
Datasets remain under their respective public licenses.