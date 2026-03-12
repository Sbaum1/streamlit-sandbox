# 🔒 LOCKED FILE (VISUAL MODULE ONLY)
# Allowed changes: append NEW render_* functions only
# Forbidden: modifying existing logic, removing traces, altering styles

# ==================================================
# FILE: features/home/forecast_viz.py
# ROLE: BASELINE FORECAST VISUALS (EXECUTIVE-GRADE)
# STATUS: LOCKED / STABLE
#
# ⚠️ GOVERNANCE:
# - This module renders visuals ONLY
# - No forecasting logic
# - No state mutation
# - No execution at import time
# ==================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.visuals import executive_line_style, confidence_band_style


# ==================================================
# SECTION 2 (LOCKED)
# BASELINE FORECAST VISUALS & TABLES
# ==================================================

def render_baseline_forecast():
    """
    Render executive-grade baseline forecast visuals.
    Requires:
      - st.session_state.latest_forecast_df
      - st.session_state.latest_forecasts
      - st.session_state.latest_model_name
    """

    df = st.session_state.get("latest_forecast_df")
    forecasts = st.session_state.get("latest_forecasts", {})
    winner = st.session_state.get("latest_model_name")

    if df is None or not forecasts or winner is None:
        st.info("Run a forecast to view executive outputs.")
        return

    st.markdown("## Executive Forecast Overview")

    fig = go.Figure()

    # ---------- Actuals ---------------------------------
    actual = st.session_state.get("committed_df")
    if isinstance(actual, pd.DataFrame) and {"date", "value"}.issubset(actual.columns):
        fig.add_trace(
            go.Scatter(
                x=actual["date"],
                y=actual["value"],
                name="Actual",
                **executive_line_style("actual"),
            )
        )

    # ---------- Forecasts (winner + comparisons) --------
    dash_cycle = ["solid", "dot", "dash", "dashdot"]

    for i, (model, series) in enumerate(forecasts.items()):
        style = executive_line_style(
            "forecast" if model == winner else "comparison"
        )
        style["line"]["dash"] = dash_cycle[i % len(dash_cycle)]

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=f"{model} forecast",
                **style,
            )
        )

    # ---------- Confidence Bands ------------------------
    if {"upper", "lower", "date"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["upper"],
                name="Confidence upper",
                **confidence_band_style("upper"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["lower"],
                name="Confidence lower",
                **confidence_band_style("lower"),
            )
        )

    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Executive Table -------------------------
    st.markdown("### Actual vs Forecast")

    required_cols = {"date", "actual", "forecast", "lower", "upper"}
    if not required_cols.issubset(df.columns):
        st.warning("Forecast table is missing required columns.")
        return

    table = df.copy()
    table["error"] = table["actual"] - table["forecast"]
    table["pct_error"] = table["error"] / table["actual"]

    st.dataframe(
        table[
            ["date", "actual", "forecast", "lower", "upper", "error", "pct_error"]
        ],
        use_container_width=True,
    )


# ==================================================
# SECTION 2A (APPENDED)
# EXECUTIVE FORECAST — UPGRADED VISUAL (V2)
# ==================================================

def render_executive_forecast_v2():
    """
    Executive-grade upgraded forecast visualization.

    APPEND-ONLY MODULE
    ------------------
    - Does NOT modify existing visuals
    - Does NOT mutate session state
    - Uses existing forecast outputs
    - Adds forward-horizon shading, model color separation,
      line hierarchy, and zero-centered scaling
    """

    df = st.session_state.get("latest_forecast_df")
    forecasts = st.session_state.get("latest_forecasts", {})
    winner = st.session_state.get("latest_model_name")
    actual = st.session_state.get("committed_df")

    if df is None or not forecasts or winner is None:
        return

    st.markdown("## Executive Forecast Chart")

    fig = go.Figure()

    # --------------------------------------------------
    # MODEL COLOR MAP (LOCAL, NON-AUTHORITATIVE)
    # --------------------------------------------------
    MODEL_COLORS = {
        "Prophet": "#4C78A8",
        "ETS": "#F58518",
        "Linear": "#E45756",
        "Naive": "#72B7B2",
        "Auto": "#9D755D",
    }

    all_y = []

    # --------------------------------------------------
    # ACTUALS
    # --------------------------------------------------
    if isinstance(actual, pd.DataFrame) and {"date", "value"}.issubset(actual.columns):
        all_y.extend(actual["value"].dropna().tolist())

        actual_style = executive_line_style("actual")
        actual_style["line"]["width"] = 2.5

        fig.add_trace(
            go.Scatter(
                x=actual["date"],
                y=actual["value"],
                name="Actual",
                **actual_style,
            )
        )

    # --------------------------------------------------
    # COMPARISON FORECASTS
    # --------------------------------------------------
    for model, series in forecasts.items():
        if model == winner:
            continue

        all_y.extend(series.dropna().tolist())

        style = executive_line_style("comparison")
        style["line"]["dash"] = "dot"
        style["line"]["width"] = 2
        style["opacity"] = 0.6

        if model in MODEL_COLORS:
            style["line"]["color"] = MODEL_COLORS[model]

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=f"{model} forecast",
                **style,
            )
        )

    # --------------------------------------------------
    # SELECTED FORECAST (PRIMARY EMPHASIS)
    # --------------------------------------------------
    selected_series = forecasts.get(winner)
    if selected_series is not None:
        all_y.extend(selected_series.dropna().tolist())

        style = executive_line_style("forecast")
        style["line"]["width"] = 4

        if winner in MODEL_COLORS:
            style["line"]["color"] = MODEL_COLORS[winner]

        fig.add_trace(
            go.Scatter(
                x=selected_series.index,
                y=selected_series.values,
                name="Selected forecast",
                **style,
            )
        )

    # --------------------------------------------------
    # CONFIDENCE BAND
    # --------------------------------------------------
    if {"date", "upper", "lower"}.issubset(df.columns):
        all_y.extend(df["upper"].dropna().tolist())
        all_y.extend(df["lower"].dropna().tolist())

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["upper"],
                **confidence_band_style("upper"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["lower"],
                fill="tonexty",
                name="Confidence band",
                **confidence_band_style("lower"),
            )
        )

    # --------------------------------------------------
    # FORWARD HORIZON SHADING
    # --------------------------------------------------
    if isinstance(actual, pd.DataFrame) and not actual.empty:
        cutoff = actual["date"].max()
        fig.add_vrect(
            x0=cutoff,
            x1=df["date"].max(),
            fillcolor="rgba(80, 120, 80, 0.15)",
            layer="below",
            line_width=0,
        )

    # --------------------------------------------------
    # ZERO-CENTERED SCALING
    # --------------------------------------------------
    if all_y:
        max_abs = max(abs(min(all_y)), abs(max(all_y)))
        fig.update_yaxes(range=[-max_abs, max_abs])

    # --------------------------------------------------
    # LAYOUT
    # --------------------------------------------------
    fig.update_layout(
        height=520,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(orientation="v"),
    )

    st.plotly_chart(fig, use_container_width=True)
