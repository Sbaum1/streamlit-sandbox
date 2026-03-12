# 🔒 LOCKED FILE (VISUAL MODULE ONLY)
# Allowed changes: this file is NEW
# Forbidden: state mutation, forecasting logic, execution at import
#
# ==================================================
# FILE: features/home/forecast_executive_v3.py
# ROLE: EXECUTIVE FORECAST CANVAS (INTERACTIVE)
# STATUS: STABLE / GOVERNANCE-COMPLIANT
#
# GOVERNANCE:
# - Visuals ONLY
# - Reads from st.session_state
# - NO state mutation
# - NO forecasting logic
# ==================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.visuals import confidence_band_style


# --------------------------------------------------
# INTERNAL HELPERS (PURE)
# --------------------------------------------------

def _compute_y_bounds(series_list, lower_q=0.02, upper_q=0.98):
    values = np.concatenate([s.values for s in series_list if s is not None])
    if values.size == 0:
        return None, None

    low = np.quantile(values, lower_q)
    high = np.quantile(values, upper_q)
    pad = (high - low) * 0.1
    return low - pad, high + pad


def _model_color_palette():
    return {
        "Auto-Blend": "#E45756",
        "Prophet": "#4C78A8",
        "ETS": "#F58518",
        "Linear": "#54A24B",
        "Naive": "#B279A2",
    }


# --------------------------------------------------
# MAIN RENDER
# --------------------------------------------------

def render_executive_forecast_v3():
    """
    Executive-grade interactive forecast visualization (v3)

    VISUAL-ONLY FEATURES:
    - Comparison model visibility
    - Selected-model isolation
    - Confidence band toggle
    - Robust Y-axis scaling
    - Zoom presets
    """

    actual = st.session_state.get("committed_df")
    forecasts = st.session_state.get("latest_forecasts", {})
    df = st.session_state.get("latest_forecast_df")
    winner = st.session_state.get("latest_model_name")

    if actual is None or not forecasts or df is None or winner is None:
        return

    st.markdown("## Forecast")

    # ==================================================
    # FORECAST FOCUS CONTROLS (LOCAL / VISUAL ONLY)
    # ==================================================
    with st.expander("Forecast Focus & Scenarios", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            show_comparisons = st.checkbox("Show comparison models", value=True)
            selected_only = st.checkbox("Selected model only", value=False)

        with col2:
            clip_extremes = st.checkbox("Robust scaling", value=True)
            show_confidence = st.checkbox("Show confidence band", value=True)

        with col3:
            zoom_preset = st.selectbox(
                "Zoom preset",
                ["Full history", "Last 36 periods", "Forecast only"],
                index=0,
            )

    fig = go.Figure()

    # ==================================================
    # ACTUALS (ANCHOR CONTEXT)
    # ==================================================
    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["value"],
            name="Actual",
            line=dict(color="#D0D0D0", width=3),
            opacity=0.9,
        )
    )

    palette = _model_color_palette()
    forecast_series_for_scaling = []

    # ==================================================
    # FORECAST MODELS
    # ==================================================
    for model, series in forecasts.items():
        color = palette.get(model, "#888888")

        if model == winner:
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=f"{model} (selected)",
                    line=dict(color=color, width=3),
                )
            )
            forecast_series_for_scaling.append(series)
            continue

        if selected_only:
            continue

        if show_comparisons:
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=model,
                    line=dict(color=color, width=2, dash="dot"),
                    opacity=0.55,
                )
            )
            forecast_series_for_scaling.append(series)

    # ==================================================
    # CONFIDENCE BAND (SAFE)
    # ==================================================
    if show_confidence and {"date", "upper", "lower"}.issubset(df.columns):
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
                **confidence_band_style("lower"),
            )
        )

    # ==================================================
    # FORECAST HORIZON SHADE
    # ==================================================
    cutoff = actual["date"].max()
    fig.add_vrect(
        x0=cutoff,
        x1=df["date"].max(),
        fillcolor="rgba(80,120,80,0.18)",
        layer="below",
        line_width=0,
    )

    # ==================================================
    # Y-AXIS SCALING
    # ==================================================
    if clip_extremes:
        low, high = _compute_y_bounds(
            [actual["value"]] + forecast_series_for_scaling
        )
        if low is not None and high is not None:
            fig.update_yaxes(range=[low, high])

    # ==================================================
    # X-AXIS ZOOM PRESETS (VALID ONLY)
    # ==================================================
    if zoom_preset == "Last 36 periods" and len(actual) >= 36:
        fig.update_xaxes(
            range=[actual["date"].iloc[-36], df["date"].max()]
        )
    elif zoom_preset == "Forecast only":
        fig.update_xaxes(range=[cutoff, df["date"].max()])

    fig.update_xaxes(rangeslider=dict(visible=True))

    # ==================================================
    # LAYOUT
    # ==================================================
    fig.update_layout(
        height=580,
        margin=dict(t=50, l=40, r=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="v"),
    )

    st.plotly_chart(fig, use_container_width=True)
