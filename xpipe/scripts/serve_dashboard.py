#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# serve_dashboard.py — X‑Pipe Streamlit dashboard (rich visuals)
# -----------------------------------------------------------------------------
# What you get
#   • Overview tab: quick KPIs, stacked token bars, cost bars
#   • Group Comparison: mean ± stdev bars with group/color/facet controls
#   • Trade‑offs: latency vs. quality scatter (per item) + per-run summary bars
#   • Calibration: reliability curve + bin table (if conf/correct present)
#   • Matrix: heatmap of quality metric (run × item)
#   • Table: filtered rows with CSV download
#
# Data sources (toggle in sidebar):
#   • output/xpipe/metrics/*.csv
#   • output/xpipe/ablations/*.csv
#
# Design notes
#   • Altair/Vega-Lite v5 safe (no YError). Error bars done via bar+rule layering.
#   • Facet charts: height set on inner layered chart; title on facet container.
#   • NA-safe: missing columns won’t crash; charts become conditional.
# =============================================================================

from __future__ import annotations
import os, glob, io
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------- Constants ---------------------------------------
METRICS_DIR = "output/xpipe/metrics"
ABLATIONS_DIR = "output/xpipe/ablations"

# ---------------------------- Altair theme ------------------------------------
def _theme():
    return {
        "config": {
            "view": {"continuousWidth": 400, "continuousHeight": 280},
            "legend": {"titleFontSize": 12, "labelFontSize": 11},
            "axis": {"labelFontSize": 11, "titleFontSize": 12, "grid": True},
            "title": {"fontSize": 16, "anchor": "start"},
            "bar": {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4}
        }
    }

alt.themes.register("xpipe", _theme)
alt.themes.enable("xpipe")

# ---------------------------- Utilities ---------------------------------------
def _safe_bool_series(s: pd.Series) -> pd.Series:
    # Avoid FutureWarning; parse loose truthy/falsy strings/nums
    def parse(v):
        sv = str(v).strip().lower()
        if sv in ("true", "1", "yes", "y", "t"): return True
        if sv in ("false", "0", "no", "n", "f"): return False
        try:
            return bool(int(sv))
        except Exception:
            return bool(v)
    return s.map(parse)

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)

@st.cache_data(show_spinner=False)
def load_metrics(include_metrics: bool=True, include_ablations: bool=True) -> pd.DataFrame:
    paths = []
    if include_metrics:
        paths += sorted(glob.glob(os.path.join(METRICS_DIR, "*.csv")))
    if include_ablations:
        paths += sorted(glob.glob(os.path.join(ABLATIONS_DIR, "*.csv")))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            run_name = os.path.splitext(os.path.basename(p))[0]
            source = "ablations" if p.startswith(ABLATIONS_DIR) else "metrics"
            df["run"] = run_name
            df["__source__"] = source
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalize booleans
    if "judge_enabled" in df.columns:
        df["judge_enabled"] = _safe_bool_series(df["judge_enabled"])

    # Normalize numerics
    _safe_numeric(df, [
        "relevance","rougeL_f","f1_token",
        "latency_ms","cost_usd",
        "synth_prompt_tokens","synth_completion_tokens",
        "judge_prompt_tokens","judge_completion_tokens",
        "conf","correct"  # optional
    ])

    # Token totals
    if "synth_prompt_tokens" in df.columns or "synth_completion_tokens" in df.columns:
        df["synth_total_tokens"] = df.get("synth_prompt_tokens", 0).fillna(0) + df.get("synth_completion_tokens", 0).fillna(0)
    if "judge_prompt_tokens" in df.columns or "judge_completion_tokens" in df.columns:
        df["judge_total_tokens"] = df.get("judge_prompt_tokens", 0).fillna(0) + df.get("judge_completion_tokens", 0).fillna(0)
    df["total_tokens_all"] = df.get("synth_total_tokens", 0).fillna(0) + df.get("judge_total_tokens", 0).fillna(0)

    # Coerce likely categoricals
    for maybe_cat in ["retriever","pipeline","item","run","__source__"]:
        if maybe_cat in df.columns:
            df[maybe_cat] = df[maybe_cat].astype(str)

    return df

# ---------------------------- Chart builders ----------------------------------
def bar_with_stdev(df, x_col, y_col, color_col=None, facet_col=None, title=""):
    base = alt.Chart(df)
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
    rules = agg.transform_filter("isValid(datum.sd_y)").mark_rule(strokeWidth=2).encode(
        x=alt.X(f"{x_col}:N"),
        y="lower:Q", y2="upper:Q",
        color=alt.Color(f"{color_col}:N", title=color_col) if color_col else alt.value("#333")
    )
    layered = (bars + rules).properties(height=320)

    if facet_col:
        return layered.facet(column=alt.Column(f"{facet_col}:N", title=facet_col)).properties(title=title)
    return layered.properties(title=title)

def tokens_stacked_bar(df, x_col="run", title="Tokens by stage (sum)"):
    if not _has_cols(df, ["synth_total_tokens","judge_total_tokens", x_col]):
        return alt.LayerChart()  # empty chart

    long = pd.melt(
        df[[x_col, "synth_total_tokens", "judge_total_tokens"]].copy(),
        id_vars=[x_col], var_name="stage", value_name="tokens"
    )
    long["stage"] = long["stage"].map({
        "synth_total_tokens": "synth",
        "judge_total_tokens": "judge"
    })
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", title=x_col),
            y=alt.Y("sum(tokens):Q", title="tokens (sum)"),
            color=alt.Color("stage:N", title="stage")
        )
        .properties(height=280, title=title)
    )

def cost_bar(df, x_col="run", title="Cost (mean per item)"):
    if not _has_cols(df, ["cost_usd", x_col]):
        return alt.LayerChart()
    return (
        alt.Chart(df)
        .mark_bar(size=28)
        .encode(
            x=alt.X(f"{x_col}:N", title=x_col),
            y=alt.Y("mean(cost_usd):Q", title="mean cost (USD)")
        )
        .properties(height=260, title=title)
    )

def latency_vs_quality(df, quality_col="relevance", color_col=None, title=""):
    if quality_col not in df.columns or "latency_ms" not in df.columns:
        return alt.LayerChart()
    enc = dict(
        x=alt.X("latency_ms:Q", title="latency (ms)"),
        y=alt.Y(f"{quality_col}:Q", title=quality_col),
        tooltip=[c for c in ["run","item","retriever","judge_enabled",quality_col,"latency_ms"] if c in df.columns]
    )
    if color_col:
        enc["color"] = alt.Color(f"{color_col}:N", title=color_col)
    return (
        alt.Chart(df.dropna(subset=["latency_ms", quality_col]))
        .mark_circle(size=80, opacity=0.6)
        .encode(**enc)
        .properties(height=360, title=title or f"{quality_col} vs latency_ms")
        .interactive()
    )

def quality_matrix(df, quality_col="relevance"):
    if not _has_cols(df, ["run","item", quality_col]):
        return alt.LayerChart()
    base = df.dropna(subset=[quality_col]).copy()
    return (
        alt.Chart(base)
        .mark_rect()
        .encode(
            x=alt.X("run:N", title="run"),
            y=alt.Y("item:N", title="item"),
            color=alt.Color(f"{quality_col}:Q", title=quality_col, scale=alt.Scale(scheme="blues"))
        )
        .properties(height=420, title=f"{quality_col} matrix: run × item")
    )

def calibration_curve(df, conf_col="conf", corr_col="correct", bins=10):
    if not _has_cols(df, [conf_col, corr_col]):
        return alt.LayerChart()
    tmp = df.dropna(subset=[conf_col, corr_col]).copy()
    if tmp.empty:
        return alt.LayerChart()

    # Clamp confidence to [0,1]
    tmp[conf_col] = tmp[conf_col].clip(0, 1)
    tmp["bin"] = pd.cut(tmp[conf_col], bins=bins, labels=False, include_lowest=True)
    agg = tmp.groupby("bin").agg(
        bin_conf=(conf_col, "mean"),
        acc=(corr_col, "mean"),
        count=(corr_col, "size")
    ).reset_index()

    points = alt.Chart(agg).mark_point(filled=True, size=80).encode(
        x=alt.X("bin_conf:Q", title="mean confidence"),
        y=alt.Y("acc:Q", title="accuracy"),
        tooltip=["bin_conf:Q","acc:Q","count:Q"]
    )
    line = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_line(strokeDash=[6,4]).encode(
        x="x:Q", y="y:Q"
    )
    bars = alt.Chart(agg).mark_bar(opacity=0.25).encode(
        x=alt.X("bin_conf:Q", title="mean confidence"),
        y=alt.Y("count:Q", title="count", axis=alt.Axis(orient="right")),
    )
    # Dual-axis via layering + resolve
    chart = alt.layer(
        bars.encode(color=alt.value("#9ecae1")),
        points.encode(color=alt.value("#08519c")),
        line.encode(color=alt.value("#6b6b6b"))
    ).resolve_scale(y='independent').properties(height=320, title="Calibration: reliability curve & bin counts")
    return chart

# ---------------------------- Page --------------------------------------------
def main():
    st.set_page_config(page_title="X‑Pipe Dashboard", layout="wide")
    st.title("X‑Pipe: Runs • Metrics • Group Comparison")

    with st.sidebar:
        st.header("Data sources")
        use_metrics = st.checkbox("Include metrics/", value=True)
        use_ablations = st.checkbox("Include ablations/", value=True)

    df = load_metrics(use_metrics, use_ablations)
    if df.empty:
        st.warning("No CSVs found in output/xpipe/metrics or output/xpipe/ablations.")
        st.stop()

    # Sidebar: filters & grouping
    with st.sidebar:
        st.header("Filters")
        runs = sorted(df["run"].unique().tolist())
        pick_all = st.checkbox("Show all runs", value=True)
        run_sel = runs if pick_all else st.multiselect("Pick runs", options=runs, default=runs[:1])
        view = df if pick_all else df[df["run"].isin(run_sel)]
        st.caption(f"{len(view)} rows · {len(view['run'].unique())} run(s)")

        st.markdown("---")
        st.header("Group comparison controls")
        compare_on = st.checkbox("Enable group comparison", value=False)
        likely_groups = [c for c in ["retriever","judge_enabled","run","__source__"] if c in view.columns]
        group_col = st.selectbox("Group (X axis)", options=likely_groups or ["run"], index=0)
        color_col = st.selectbox("Color (optional)", options=["(none)"] + likely_groups, index=0)
        color_col = None if color_col == "(none)" else color_col
        facet_col = st.selectbox("Facet (optional)", options=["(none)"] + likely_groups, index=0)
        facet_col = None if facet_col == "(none)" else facet_col

        st.markdown("---")
        st.header("Metrics")
        quality_candidates = [c for c in ["relevance","rougeL_f","f1_token"] if c in view.columns]
        eff_candidates = [c for c in ["latency_ms","total_tokens_all","synth_total_tokens","judge_total_tokens","cost_usd"] if c in view.columns]
        y_quality = st.selectbox("Quality metric", options=quality_candidates or ["relevance"])
        y_eff = st.selectbox("Efficiency metric", options=eff_candidates or ["latency_ms"])

    # Tabs
    t_over, t_group, t_trade, t_calib, t_matrix, t_table = st.tabs(
        ["Overview", "Group Comparison", "Trade‑offs", "Calibration", "Matrix", "Table"]
    )

    # -------------------- Overview --------------------
    with t_over:
        c1, c2, c3 = st.columns([2,2,2])
        with c1:
            st.subheader("Tokens")
            st.altair_chart(tokens_stacked_bar(view, x_col="run", title="Tokens by stage (sum)"), use_container_width=True)
        with c2:
            st.subheader("Latency")
            st.altair_chart(bar_with_stdev(view, "run", "latency_ms", color_col=None, facet_col=None,
                                           title="Mean ± stdev latency (ms) by run"), use_container_width=True)
        with c3:
            st.subheader("Cost")
            st.altair_chart(cost_bar(view, x_col="run", title="Mean cost per item (USD)"), use_container_width=True)

        st.markdown("### Quick KPIs")
        agg_cols = [c for c in ["relevance","rougeL_f","f1_token","latency_ms","total_tokens_all","cost_usd"] if c in view.columns]
        if agg_cols:
            run_agg = view.groupby("run")[agg_cols].mean(numeric_only=True).reset_index().sort_values("run")
            st.dataframe(run_agg, use_container_width=True)
        else:
            st.info("No numeric metrics to summarize.")

    # -------------------- Group Comparison --------------------
    with t_group:
        st.subheader("Quality")
        chq = bar_with_stdev(
            view,
            x_col=group_col if compare_on else "run",
            y_col=y_quality,
            color_col=color_col,
            facet_col=facet_col,
            title=f"Mean ± stdev of {y_quality}" + ("" if compare_on else " by run")
        )
        st.altair_chart(chq, use_container_width=True)

        st.subheader("Efficiency")
        che = bar_with_stdev(
            view,
            x_col=group_col if compare_on else "run",
            y_col=y_eff,
            color_col=color_col,
            facet_col=facet_col,
            title=f"Mean ± stdev of {y_eff}" + ("" if compare_on else " by run")
        )
        st.altair_chart(che, use_container_width=True)

    # -------------------- Trade-offs --------------------
    with t_trade:
        st.subheader("Latency vs. quality (per item)")
        st.altair_chart(
            latency_vs_quality(view, quality_col=y_quality, color_col=color_col),
            use_container_width=True
        )
        st.caption("Tip: Use color=judge_enabled to see judge ablations, or color=retriever to see retriever swaps.")

        st.subheader("Per‑run quality vs. latency")
        # two side-by-side bars (mean quality, mean latency)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.altair_chart(
                bar_with_stdev(view, "run", y_quality, color_col=None, facet_col=None,
                               title=f"Mean ± stdev {y_quality} by run"),
                use_container_width=True
            )
        with cc2:
            st.altair_chart(
                bar_with_stdev(view, "run", "latency_ms", color_col=None, facet_col=None,
                               title="Mean ± stdev latency (ms) by run"),
                use_container_width=True
            )

    # -------------------- Calibration --------------------
    with t_calib:
        st.subheader("Calibration")
        if _has_cols(view, ["conf","correct"]):
            st.altair_chart(calibration_curve(view, "conf", "correct"), use_container_width=True)
            # Also show the bins
            tmp = view.dropna(subset=["conf","correct"]).copy()
            tmp["conf"] = tmp["conf"].clip(0,1)
            tmp["bin"] = pd.cut(tmp["conf"], bins=10, labels=False, include_lowest=True)
            bins = tmp.groupby("bin").agg(
                mean_conf=("conf","mean"),
                acc=("correct","mean"),
                count=("correct","size")
            ).reset_index()
            st.dataframe(bins, use_container_width=True)
        else:
            st.info("No calibration columns found. Log `conf` (∈[0,1]) and `correct` (0/1) to enable this view.")

    # -------------------- Matrix --------------------
    with t_matrix:
        st.subheader("Quality matrix (run × item)")
        st.altair_chart(quality_matrix(view, quality_col=y_quality), use_container_width=True)

    # -------------------- Table --------------------
    with t_table:
        st.subheader("Filtered rows")
        st.dataframe(view, use_container_width=True, height=480)
        st.download_button(
            "Download filtered CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="xpipe_filtered_metrics.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()