# attribution.py — stub
# SPDX-License-Identifier: BSD-3-Clause
# xpipe/attribution.py
# ----------------------------------------------------------------------
# XPipe Ablations / Factor Sweeps
# ----------------------------------------------------------------------
# This module provides a tiny helper to run *factorial sweeps* (ablations)
# over a pipeline function. It’s intentionally minimal and dependency-light:
#
#   • _grid(factors):     Cartesian product over named factors.
#   • ablate(...):        Evaluate pipeline_fn(item, **variant) across
#                         all factor combinations and dataset items.
#                         Returns a tidy pandas DataFrame for analysis.
#
# Typical Use:
#   def run_variant(example, temp, top_p):
#       # call your pipeline with the variant knobs
#       out = pipeline(example, temperature=temp, top_p=top_p)
#       return {"metric": out["score"], "latency_ms": out["latency"]}
#
#   ds = ["q1", "q2", "q3"]
#   factors = {"temp": [0.0, 0.5], "top_p": [0.8, 0.95]}
#   df = ablate(run_variant, ds, factors)
#   # df columns: item, temp, top_p, metric, latency_ms, ...
#
# Notes:
#   • `merge_result` allows you to normalize arbitrary return values into
#     a dict of columns if your pipeline returns non-dict types.
#   • The resulting DataFrame is "long" format—ideal for groupby/plotting.
# ----------------------------------------------------------------------

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


def _grid(factors: Mapping[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """
    Build a Cartesian product over named factor lists.

    Parameters
    ----------
    factors : Mapping[str, Iterable[Any]]
        A dict-like mapping from factor name to a sequence of values, e.g.:
          {"temperature": [0.0, 0.5], "top_p": [0.8, 0.95]}

    Yields
    ------
    Dict[str, Any]
        One dict per combination, e.g. {"temperature": 0.0, "top_p": 0.8}.
    """
    keys = list(factors.keys())
    values = [list(v) for v in factors.values()]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def ablate(
    pipeline_fn: Callable[..., Any],
    dataset: Sequence[Any],
    factors: Mapping[str, Iterable[Any]],
    *,
    merge_result: Callable[[Any], Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Run a simple factorial sweep over `factors`, evaluating
    `pipeline_fn(item, **variant)` for each item and variant.

    This function is agnostic to your pipeline; it only assumes that
    `pipeline_fn` can be called with the dataset item as the first argument,
    and that factor values can be passed as keyword arguments.

    Parameters
    ----------
    pipeline_fn : Callable[..., Any]
        Function with signature like: (item, **variant) -> dict | Any
        • If it returns a dict, keys are merged into the output row.
        • Otherwise, the value is stored under the column "output".
        • For custom normalization, use `merge_result`.
    dataset : Sequence[Any]
        Items to evaluate (e.g., questions, inputs, sample IDs).
    factors : Mapping[str, Iterable[Any]]
        Factor grid to sweep, e.g. {"temperature": [0.0, 0.5], "top_p": [0.8, 0.95]}.
    merge_result : Optional[Callable[[Any], Dict[str, Any]]], default=None
        Optional converter to map arbitrary pipeline outputs to a dict of
        columns (e.g., extracting metrics, latencies, or flags).

    Returns
    -------
    pandas.DataFrame
        Long-form table with one row per (variant × item). Columns include:
          • "item" (the dataset element)
          • each factor name (e.g., "temperature", "top_p")
          • either:
              - keys from the dict returned by `pipeline_fn`, or
              - "output" if `pipeline_fn` returned a non-dict, or
              - keys from `merge_result(out)` when provided.

    Examples
    --------
    >>> def run_variant(x, temp, top_p):
    ...     y = my_pipeline(x, temperature=temp, top_p=top_p)
    ...     return {"score": y["score"], "ok": y["ok"]}
    >>> df = ablate(run_variant, ["q1", "q2"], {"temp": [0.0, 0.5], "top_p": [0.8, 0.95]})
    >>> df.head()
       item  temp  top_p  score     ok
    0   q1   0.0   0.80   0.71   True
    1   q1   0.0   0.95   0.69   True
    2   q1   0.5   0.80   0.76   True
    3   q1   0.5   0.95   0.73   True
    4   q2   0.0   0.80   0.66   True
    """
    rows: List[Dict[str, Any]] = []
    for variant in _grid(factors):
        for item in dataset:
            out = pipeline_fn(item, **variant)
            row: Dict[str, Any] = {"item": item, **variant}
            if merge_result is not None:
                row.update(merge_result(out))
            elif isinstance(out, dict):
                row.update(out)
            else:
                row["output"] = out
            rows.append(row)
    return pd.DataFrame(rows)