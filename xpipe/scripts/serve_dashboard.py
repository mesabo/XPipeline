#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# serve_dashboard.py — X‑Pipe Streamlit dashboard (group comparison ready)
# ----------------------------------------------------------------------
# What you get
# - Run picker (from output/xpipe/metrics/*.csv)
# - Single-run and multi-run views
# - Group comparison (e.g., retriever × judge_enabled) with bar+stdev whiskers
# - Quality & Efficiency metrics, token/cost summaries
#
# Notes
# - No sudo/admin needed. Works in conda env with: streamlit, pandas, altair
# - Error bars use a robust rule+bar layering (no YError schema params)
# ======================================================================

import os
import glob
import json
import math
import pandas as pd
import streamlit as st
import altair as alt

METRICS_DIR = "output/xpipe/metrics"

# ----------------------------- Helpers --------------------------------
def load_all_metrics(metrics_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(metrics_dir, "*.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            run_name = os.path.splitext(os.path.basename(f))[0]
            df["run"] = run_name
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed reading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Best-effort normalization of expected columns
    # If your metrics add columns, they'll just carry through.
    expected_bools = ["judge_enabled"]
    for col in expected_bools:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"True": True, "False": False}).astype(bool)

    # Fill missing numeric columns so charts don’t break
    numeric_maybe = [
        "relevance", "rougeL_f", "f1_token",
        "latency_ms", "cost_usd",
        "synth_prompt_tokens","synth_completion_tokens",
        "judge_prompt_tokens","judge_completion_tokens",
    ]
    for col in numeric_maybe:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convenience sums
    if "synth_prompt_tokens" in df.columns or "synth_completion_tokens" in df.columns:
        df["synth_total_tokens"] = df.get("synth_prompt_tokens", 0).fillna(0) + df.get("synth_completion_tokens", 0).fillna(0)
    if "judge_prompt_tokens" in df.columns or "judge_completion_tokens" in df.columns:
        df["judge_total_tokens"] = df.get("judge_prompt_tokens", 0).fillna(0) + df.get("judge_completion_tokens", 0).fillna(0)
    df["total_tokens_all"] = df.get("synth_total_tokens", 0).fillna(0) + df.get("judge_total_tokens", 0).fillna(0)

    # Common categorical hints if present
    for maybe_cat in ["retriever", "pipeline", "item"]:
        if maybe_cat in df.columns:
            df[maybe_cat] = df[maybe_cat].astype(str)

    return df


def bar_with_stdev(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None,
                   facet_col: str | None, title: str):
    """
    Robust mean bar + stdev whiskers using aggregate+calculate+layer (Altair/Vega-Lite 5 friendly).
    No YError schema fields are used.

    IMPORTANT: When using facet(), the top-level FacetChart cannot receive 'height'.
               We attach 'height' to the inner layered chart, and 'title' to the facet container.
    """
    base = alt.Chart(df)

    # Aggregate mean and stdev per grouping
    group_cols = [c for c in [x_col, color_col, facet_col] if c]
    agg = base.transform_aggregate(
        mean_y=f"mean({y_col})",
        sd_y=f"stdev({y_col})",
        groupby=group_cols
    ).transform_calculate(
        lower="datum.mean_y - (isValid(datum.sd_y) ? datum.sd_y : 0)",
        upper="datum.mean_y + (isValid(datum.sd_y) ? datum.sd_y : 0)"
    )

    enc = dict(
        x=alt.X(f"{x_col}:N", title=x_col),
        y=alt.Y("mean_y:Q", title=f"mean({y_col})")
    )
    if color_col:
        enc["color"] = alt.Color(f"{color_col}:N", title=color_col)

    bars = agg.mark_bar(size=28).encode(**enc)

    # Whiskers via rules (y=lower, y2=upper). Hide if sd is null.
    rules = agg.transform_filter("isValid(datum.sd_y)").mark_rule(strokeWidth=2).encode(
        x=alt.X(f"{x_col}:N"),
        y="lower:Q",
        y2="upper:Q",
        color=alt.Color(f"{color_col}:N", title=color_col) if color_col else alt.value("#333")
    )

    layered = (bars + rules).properties(height=320)  # height on inner chart only

    if facet_col:
        # Put the title on the facet container; DO NOT set height here.
        return layered.facet(column=alt.Column(f"{facet_col}:N", title=facet_col)).properties(title=title)
    else:
        # Non-faceted chart can carry both title and height here
        return layered.properties(title=title)

def two_axis_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None, tooltip_cols: list[str], title: str):
    enc = dict(
        x=alt.X(f"{x_col}:Q", title=x_col),
        y=alt.Y(f"{y_col}:Q", title=y_col),
        tooltip=[c if (":" in c) else f"{c}:N" for c in tooltip_cols],
    )
    if color_col:
        enc["color"] = alt.Color(f"{color_col}:N", title=color_col)

    return alt.Chart(df).mark_circle(size=80, opacity=0.6).encode(**enc).properties(
        height=380, title=title
    ).interactive()


def safe_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


# ----------------------------- UI -------------------------------------
def main():
    st.set_page_config(page_title="X‑Pipe Dashboard", layout="wide")
    st.title("X‑Pipe: Runs • Metrics • Group Comparison")

    df = load_all_metrics(METRICS_DIR)
    if df.empty:
        st.warning(f"No metrics found in {METRICS_DIR}. Run a pipeline first.")
        return

    with st.sidebar:
        st.header("Controls")

        # Run selection
        runs = sorted(df["run"].unique().tolist())
        all_runs = st.checkbox("Show all runs", value=True)
        picked_runs = runs if all_runs else st.multiselect("Pick runs", options=runs, default=runs[:1])

        df_view = df if all_runs else df[df["run"].isin(picked_runs)].copy()
        st.markdown("---")

        # Comparison mode
        st.subheader("Group Comparison")
        compare_on = st.checkbox("Enable group comparison", value=False)

        # Grouping columns available (cat-like)
        likely_group_cols = [c for c in ["retriever", "judge_enabled", "run"] if c in df_view.columns]
        group_col = st.selectbox("Group (X axis)", options=likely_group_cols or ["run"], index=0)
        color_col = st.selectbox("Color (optional)", options=["(none)"] + likely_group_cols, index=0)
        color_col = None if color_col == "(none)" else color_col
        facet_col = st.selectbox("Facet (optional)", options=["(none)"] + likely_group_cols, index=0)
        facet_col = None if facet_col == "(none)" else facet_col

        st.markdown("---")
        st.subheader("Metrics")

        quality_candidates = safe_cols(df_view, ["relevance", "rougeL_f", "f1_token"])
        eff_candidates = safe_cols(df_view, ["latency_ms", "total_tokens_all", "synth_total_tokens", "judge_total_tokens", "cost_usd"])

        y_quality = st.selectbox("Quality metric", options=quality_candidates or ["relevance"])
        y_eff = st.selectbox("Efficiency metric", options=eff_candidates or ["latency_ms"])

    # ---------------- Main panels ----------------
    st.markdown("### Dataset")
    st.caption(f"Loaded {len(df_view)} rows from {len(df_view['run'].unique())} run(s).")

    # Quality chart(s)
    st.markdown("### Quality")
    if compare_on:
        chq = bar_with_stdev(df_view, x_col=group_col, y_col=y_quality, color_col=color_col, facet_col=facet_col,
                             title=f"Mean ± stdev of {y_quality}")
        st.altair_chart(chq, use_container_width=True)
    else:
        # per-run bar
        chq = bar_with_stdev(df_view, x_col="run", y_col=y_quality, color_col=color_col, facet_col=facet_col,
                             title=f"Mean ± stdev of {y_quality} by run")
        st.altair_chart(chq, use_container_width=True)

    # Efficiency chart(s)
    st.markdown("### Efficiency")
    if compare_on:
        che = bar_with_stdev(df_view, x_col=group_col, y_col=y_eff, color_col=color_col, facet_col=facet_col,
                             title=f"Mean ± stdev of {y_eff}")
        st.altair_chart(che, use_container_width=True)
    else:
        che = bar_with_stdev(df_view, x_col="run", y_col=y_eff, color_col=color_col, facet_col=facet_col,
                             title=f"Mean ± stdev of {y_eff} by run")
        st.altair_chart(che, use_container_width=True)

    # Scatter: latency vs quality per item (good for seeing trade-offs)
    st.markdown("### Trade‑off: Latency vs Quality (per query/item)")
    lat_col = "latency_ms" if "latency_ms" in df_view.columns else None
    if lat_col:
        tooltip = safe_cols(df_view, ["run", "item", "retriever", "judge_enabled", y_quality, "latency_ms"])
        chs = two_axis_scatter(df_view.dropna(subset=[y_quality, lat_col]),
                               x_col=lat_col, y_col=y_quality, color_col=color_col,
                               tooltip_cols=tooltip,
                               title=f"{y_quality} vs {lat_col}")
        st.altair_chart(chs, use_container_width=True)
    else:
        st.info("No latency column found; skipping scatter.")

    # Table: per-run aggregates
    st.markdown("### Per‑run summary")
    agg_cols = safe_cols(df_view, ["relevance","rougeL_f","f1_token","latency_ms","total_tokens_all","synth_total_tokens","judge_total_tokens","cost_usd"])
    if agg_cols:
        run_agg = df_view.groupby("run")[agg_cols].mean(numeric_only=True).reset_index()
        run_agg = run_agg.sort_values("run")
        st.dataframe(run_agg, use_container_width=True)
    else:
        st.info("No numeric metrics to summarize.")

    # Raw rows
    with st.expander("Raw rows (filtered)", expanded=False):
        st.dataframe(df_view, use_container_width=True)


if __name__ == "__main__":
    main()