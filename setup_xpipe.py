#!/usr/bin/env python3
import os

# Define folder + file structure
structure = {
    "xpipe/": [
        "main.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
    ],
    "xpipe/xpipe/": [
        "__init__.py",
        "trace.py",
        "metrics.py",
        "attribution.py",
    ],
    "xpipe/xpipe/runners/": [
        "__init__.py",
        "rag.py",
        "multi_agent.py",
        "vision_text.py",
    ],
    "xpipe/xpipe/dashboard/": [
        "__main__.py",
    ],
    "xpipe/configs/": [
        "experiment_rag.yaml",
        "models.yaml",
    ],
    "xpipe/scripts/": [
        "run_xpipe.sbatch",
    ],
    "xpipe/output/xpipe/": [
        # Just ensure folders exist, no files
    ],
    "xpipe/output/xpipe/runs/": [],
    "xpipe/output/xpipe/metrics/": [],
    "xpipe/output/xpipe/ablations/": [],
    "xpipe/output/xpipe/figs/": [],
    "xpipe/output/artifacts/": [],
}

# Create directories and empty files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for f in files:
        path = os.path.join(folder, f)
        with open(path, "w", encoding="utf-8") as fh:
            if f.endswith(".py"):
                fh.write("# " + f + " — stub\n")
            elif f.endswith(".yaml"):
                fh.write("# " + f + " — config stub\n")
            elif f.endswith(".sbatch"):
                fh.write("#!/usr/bin/env bash\n# SLURM run script stub\n")
            elif f == "README.md":
                fh.write("# X-Pipe\n\nExplainable Evaluation Framework for LLM Pipelines\n")
            elif f == "requirements.txt":
                fh.write("# requirements for X-Pipe\npandas\nyaml\n")
            elif f == "LICENSE":
                fh.write("MIT License\n")

print("✅ X-Pipe module structure created.")