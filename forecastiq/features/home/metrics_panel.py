# ==================================================
# FILE: features/home/metrics_panel.py
# ROLE: MODEL METRICS + RUN COMPARISON (LOCKED)
# ==================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_model_metrics():
    metrics = st.session_state.get("latest_metrics")
    if isinstance(metrics, pd.DataFrame):
        st.markdown("### Model Performance Comparison")
        st.dataframe(metrics.sort_values("RMSE"), use_container_width=True)


def render_run_comparison():
    history = st.session_state.get("forecast_history", [])
    if len(history) < 2:
        return

    st.markdown("## Forecast Comparison (Run vs Run)")

    labels = [f"{h['run_id']} — {h['model']}" for h in history]
    a = st.selectbox("Run A", labels, index=len(labels) - 2)
    b = st.selectbox("Run B", labels, index=len(labels) - 1)

    ra = history[labels.index(a)]
    rb = history[labels.index(b)]

    merged = ra["forecast_df"].merge(
        rb["forecast_df"], on="date", suffixes=("_A", "_B")
    )
    merged["delta"] = merged["forecast_B"] - merged["forecast_A"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["forecast_A"], name=a))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["forecast_B"], name=b))

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        merged[["date", "forecast_A", "forecast_B", "delta"]],
        use_container_width=True,
    )
