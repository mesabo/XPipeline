# trace.py — stub
# SPDX-License-Identifier: BSD-3-Clause
# xpipe/trace.py
# ----------------------------------------------------------------------
# XPipe Tracing Utility
# ----------------------------------------------------------------------
# Provides lightweight, dependency-free tracing for multi-stage pipelines.
#
# Features:
#   • span(stage) context manager
#       - automatically logs "start" / "end" events
#       - allows intermediate .log(data) calls
#   • token usage accounting (prompt/completion counts)
#   • artifact logging (strings, dicts, file paths, etc.)
#   • persistence to JSONL (one line per run for robustness)
#
# Intended Usage:
#   trace = Trace(experiment="rag", run_tags={"pipeline": "demo"})
#   with trace.span("retrieve") as sp:
#       sp.log({"docs": 3})
#   trace.add_tokens(prompt=42, completion=17)
#   trace.log_artifact("synthesize", "answer", "Paris is the capital of France")
#   run_path = trace.save()
#
# Output Schema (JSONL record):
# {
#   "experiment": "...",
#   "run_id": "...",
#   "run_tags": {...},
#   "started_ts_ms": <epoch ms>,
#   "wall_time_ms": <elapsed>,
#   "token_usage": {"prompt": ..., "completion": ...},
#   "events": [
#       {"ts_ms": ..., "stage": "...", "type": "start"},
#       {"ts_ms": ..., "stage": "...", "type": "log", "data": {...}},
#       {"ts_ms": ..., "stage": "...", "type": "end", "data": {"latency_ms": ...}},
#       {"ts_ms": ..., "stage": "...", "type": "artifact", "data": {"name": ..., "value": ...}},
#       ...
#   ]
# }
# ----------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from contextlib import contextmanager


# ----------------------------------------------------------------------
# Internal Helpers
# ----------------------------------------------------------------------
def _now_ms() -> int:
    """Current wall-clock time in milliseconds since epoch."""
    return int(time.time() * 1000)


@dataclass
class Event:
    """Represents a single trace event (start/log/end/artifact)."""
    ts_ms: int
    stage: str
    type: str  # "start" | "log" | "end" | "artifact"
    data: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Trace Class
# ----------------------------------------------------------------------
class Trace:
    """
    Lightweight trace logger for XPipe runs.

    Typical lifecycle:
      1. Initialize Trace with experiment name + optional run tags.
      2. Use .span(stage) to measure sections of the pipeline.
      3. Log artifacts (answers, metadata, paths).
      4. Track token usage.
      5. Save to JSONL at end.

    Compatible with downstream analysis, dashboards, or pretty-printing.
    """

    def __init__(
        self,
        experiment: str,
        run_tags: Optional[Dict[str, Any]] = None,
        logdir: str = "output/xpipe",
        run_id: Optional[str] = None,
    ) -> None:
        self.experiment = experiment
        self.run_tags = run_tags or {}
        self.logdir = logdir
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self._start_ms = _now_ms()
        self._events: List[Event] = []
        self._token_usage = {"prompt": 0, "completion": 0}
        os.makedirs(self._runs_dir, exist_ok=True)

    # ---------- properties ----------
    @property
    def wall_time_ms(self) -> int:
        """Elapsed wall time (ms) since Trace creation."""
        return max(0, _now_ms() - self._start_ms)

    @property
    def token_usage(self) -> Dict[str, int]:
        """Return a copy of accumulated token usage."""
        return dict(self._token_usage)

    @property
    def run_path(self) -> str:
        """Default filename for saving this run (timestamped JSONL)."""
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"{stamp}_{self.experiment}_{self.run_id}.jsonl"
        return os.path.join(self._runs_dir, fname)

    @property
    def _runs_dir(self) -> str:
        """Directory under logdir where run JSONLs are stored."""
        return os.path.join(self.logdir, "runs")

    # ---------- core API ----------
    def add_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        """Accumulate token usage counts for prompt and completion."""
        self._token_usage["prompt"] += int(prompt)
        self._token_usage["completion"] += int(completion)

    def log_artifact(self, stage: str, name: str, value: Any) -> None:
        """
        Log an artifact (any serializable value) produced at a given stage.
        Example: final answers, retrieved docs, intermediate JSON blobs.
        """
        self._events.append(
            Event(ts_ms=_now_ms(), stage=stage, type="artifact", data={"name": name, "value": value})
        )

    @contextmanager
    def span(self, stage: str):
        """
        Context manager for timing/logging a pipeline stage.

        Usage:
            with trace.span("retrieve") as sp:
                sp.log({"num_docs": 5})
        """
        t0 = _now_ms()
        self._events.append(Event(ts_ms=t0, stage=stage, type="start"))

        class _Span:
            def log(_self, data: Dict[str, Any]) -> None:
                self._events.append(Event(ts_ms=_now_ms(), stage=stage, type="log", data=data))

        try:
            yield _Span()
        finally:
            t1 = _now_ms()
            self._events.append(Event(ts_ms=t1, stage=stage, type="end", data={"latency_ms": t1 - t0}))

    # ---------- persistence ----------
    def to_record(self) -> Dict[str, Any]:
        """Convert full trace state into a serializable dict record."""
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "run_tags": self.run_tags,
            "started_ts_ms": self._start_ms,
            "wall_time_ms": self.wall_time_ms,
            "token_usage": self._token_usage,
            "events": [asdict(e) for e in self._events],
        }

    def save(self, path: Optional[str] = None) -> str:
        """
        Save trace to disk as a JSONL record.
        Returns final file path.
        """
        path = path or self.run_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_record(), ensure_ascii=False) + "\n")
        return path