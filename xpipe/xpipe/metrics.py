# metrics.py — stub
# SPDX-License-Identifier: BSD-3-Clause
# xpipe/metrics.py
# ----------------------------------------------------------------------
# XPipe Metrics Utility
# ----------------------------------------------------------------------
# Provides a lightweight abstraction for collecting, storing, and saving
# per-run evaluation metrics.
#
# Features:
#   • Append metric rows incrementally during pipeline execution.
#   • Export accumulated metrics to a pandas DataFrame.
#   • Persist metrics as CSV (for further analysis/plots).
#
# Typical Usage:
#   metrics = MetricLog()
#   metrics.add(pipeline="rag", item="q1", relevance=0.82, faithfulness=0.76)
#   metrics.add(pipeline="rag", item="q2", relevance=0.90, faithfulness=0.81)
#   df = metrics.to_dataframe()
#   path = metrics.save("output/xpipe/metrics/demo.csv")
#
# CSV Output:
#   pipeline,item,relevance,faithfulness,latency_ms,cost_usd,...
#   rag,q1,0.82,0.76,123,0.0
#   rag,q2,0.90,0.81,110,0.0
# ----------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd


class MetricLog:
    """
    Unified metrics sink (task + system).

    Methods
    -------
    add(**kwargs):
        Append one row of metrics (arbitrary key-value pairs).
        Example: add(pipeline="rag", item="q1", relevance=0.75)

    to_dataframe() -> pd.DataFrame:
        Convert collected metrics to a pandas DataFrame.

    save(path: str) -> str:
        Save collected metrics to CSV at given path.
        Returns final file path.
    """

    def __init__(self) -> None:
        # Internal buffer of metrics rows (each is a dict)
        self._rows: List[Dict[str, Any]] = []

    # ---- write API ----
    def add(self, **kwargs: Any) -> None:
        """
        Append one row of metrics.

        Parameters
        ----------
        **kwargs : Any
            Arbitrary metrics fields (e.g., pipeline="rag", relevance=0.8).
        """
        self._rows.append(dict(kwargs))

    # ---- read/serialize ----
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected metrics to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per call to .add().
        """
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

    def save(self, path: str = "output/xpipe/metrics/metrics.csv") -> str:
        """
        Save collected metrics to a CSV file.

        Parameters
        ----------
        path : str, optional
            Destination path (default: "output/xpipe/metrics/metrics.csv").

        Returns
        -------
        str
            The path where the CSV was written.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path