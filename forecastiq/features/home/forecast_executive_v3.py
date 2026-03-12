# 🔒 LOCKED FILE (VISUAL MODULE ONLY)
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

from utils.visuals import executive_line_style, confidence_band_style


# --------------------------------------------------
# INTERNAL HELPERS (PURE)
# --------------------------------------------------

def _compute_y_bounds(series_list, lower_q=0.02, upper_q=0.98):
    values = np.concatenate([s.values for s in series_list if s is not None])
    if len(values) == 0:
        return None, None

    low = np.quantile(values, lower_q)
    high = np.quantile(values, upper_q)
    pad = (high - low) * 0.10
    return low - pad, high + pad


def _model_color_palette():
    return {
        "Auto-Blend": "#E45756",
        "Prophet": "#4C78A8",
        "ETS": "#F58518",
        "Linear": "#54A24B",
        "Naive": "#B279A2",
        "ARIMA": "#72B7B2",
    }


# --------------------------------------------------
# MAIN RENDER
# --------------------------------------------------

def render_executive_forecast_v3():
    """
    Executive Forecast Focus canvas (visual-only)

    PURPOSE:
    - Focused forecast inspection
    - Model comparison
    - Confidence framing
    - Scenario overlay (deterministic, optional)
    """

    actual = st.session_state.get("committed_df")
    forecasts = st.session_state.get("latest_forecasts", {})
    df = st.session_state.get("latest_forecast_df")
    winner = st.session_state.get("latest_model_name")

    # Scenario state (optional)
    scenario_df = st.session_state.get("scenario_forecast_df")
    scenario_id = st.session_state.get("active_scenario_id")
    scenario_audit = st.session_state.get("active_scenario_audit")

    if actual is None or not forecasts or df is None or winner is None:
        return

    # ==================================================
    # TITLE
    # ==================================================
    st.markdown("## Forecast")

    # ==================================================
    # FORECAST FOCUS & SCENARIOS (COLLAPSED BY DEFAULT)
    # ==================================================
    with st.expander("Forecast Focus & Scenarios", expanded=False):

        c1, c2, c3 = st.columns(3)

        with c1:
            model_visibility = st.radio(
                "Model visibility",
                ["All models", "Selected + comparisons", "Selected only"],
                index=1,
            )

        with c2:
            confidence_mode = st.radio(
                "Confidence framing",
                ["Off", "Light", "Full"],
                index=2,
            )

        with c3:
            zoom_preset = st.selectbox(
                "Zoom preset",
                ["Full history", "Last 36 periods", "Forecast only"],
            )

        st.divider()

        st.markdown("**Scenario preview (visual only)**")

        s1, s2, s3 = st.columns(3)

        with s1:
            st.slider("Level shift (%)", -20.0, 20.0, 0.0, 0.5)

        with s2:
            st.slider("Trend adjustment (%)", -20.0, 20.0, 0.0, 0.5)

        with s3:
            st.slider("Shock impact (%)", -30.0, 30.0, 0.0, 0.5)

        st.slider(
            "Shock timing (forecast horizon)",
            0.0,
            1.0,
            0.50,
            0.05,
        )

    # ==================================================
    # FIGURE
    # ==================================================
    fig = go.Figure()

    # ------------------------------
    # ACTUALS
    # ------------------------------
    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["value"],
            name="Actual",
            line=dict(color="#E0E0E0", width=3),
            opacity=0.9,
        )
    )

    palette = _model_color_palette()
    forecast_series_for_scaling = []

    # ------------------------------
    # BASELINE FORECASTS
    # ------------------------------
    for model, series in forecasts.items():
        color = palette.get(model, "#888888")
        is_selected = model == winner

        if model_visibility == "Selected only" and not is_selected:
            continue

        if model_visibility == "Selected + comparisons" and not is_selected:
            style = dict(color=color, width=2, dash="dot")
            opacity = 0.55
        elif is_selected:
            style = dict(color=color, width=3)
            opacity = 1.0
        else:
            style = dict(color=color, width=2, dash="dot")
            opacity = 0.6

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=f"{model}" + (" (selected)" if is_selected else ""),
                line=style,
                opacity=opacity,
            )
        )

        forecast_series_for_scaling.append(series)

    # ------------------------------
    # SCENARIO OVERLAY (OPTIONAL)
    # ------------------------------
    if (
        scenario_df is not None
        and scenario_id
        and scenario_id != "baseline"
        and {"date", "forecast", "is_future"}.issubset(scenario_df.columns)
    ):
        future = scenario_df[scenario_df["is_future"]]

        if not future.empty:
            scenario_name = scenario_audit.get("scenario_name", "Scenario") if scenario_audit else "Scenario"
            color = palette.get(winner, "#E45756")

            fig.add_trace(
                go.Scatter(
                    x=future["date"],
                    y=future["forecast"],
                    name=f"Scenario: {scenario_name}",
                    line=dict(color=color, width=3, dash="dash"),
                    opacity=0.9,
                )
            )

            forecast_series_for_scaling.append(future["forecast"])

    # ------------------------------
    # CONFIDENCE BANDS
    # ------------------------------
    if confidence_mode != "Off" and {"date", "upper", "lower"}.issubset(df.columns):
        alpha = 0.15 if confidence_mode == "Light" else 0.30

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["upper"],
                **confidence_band_style("upper"),
                opacity=alpha,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["lower"],
                fill="tonexty",
                **confidence_band_style("lower"),
                opacity=alpha,
            )
        )

    # ------------------------------
    # FORECAST HORIZON SHADE
    # ------------------------------
    cutoff = actual["date"].max()
    fig.add_vrect(
        x0=cutoff,
        x1=df["date"].max(),
        fillcolor="rgba(80,120,80,0.18)",
        layer="below",
        line_width=0,
    )

    # ------------------------------
    # Y-AXIS SCALING
    # ------------------------------
    low, high = _compute_y_bounds([actual["value"]] + forecast_series_for_scaling)
    if low is not None and high is not None:
        fig.update_yaxes(range=[low, high])

    # ------------------------------
    # X-AXIS CONTROL
    # ------------------------------
    if zoom_preset == "Last 36 periods" and len(actual) >= 36:
        fig.update_xaxes(range=[actual["date"].iloc[-36], df["date"].max()])
    elif zoom_preset == "Forecast only":
        fig.update_xaxes(range=[cutoff, df["date"].max()])

    fig.update_xaxes(rangeslider=dict(visible=True))

    # ------------------------------
    # LAYOUT
    # ------------------------------
    fig.update_layout(
        height=600,
        margin=dict(t=50, l=40, r=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="v"),
    )

    st.plotly_chart(fig, use_container_width=True)
