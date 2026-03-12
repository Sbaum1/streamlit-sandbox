# ==================================================
# FILE: features/home/analytics_surface.py
# ROLE: MODEL METRICS + RUN COMPARISON (LOCKED)
# STATUS: EXECUTIVE-GRADE / STABLE
#
# GOVERNANCE:
# - Visuals ONLY
# - Reads from st.session_state
# - NO state mutation
# - NO forecasting logic
# ==================================================

import streamlit as st
import pandas as pd


# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

def render_model_metrics():
    metrics = st.session_state.get("latest_metrics")
    metric_priority = st.session_state.get("metric_priority", "RMSE")

    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return

    st.markdown("### Model Performance Comparison")

    df = metrics.copy()

    # Defensive: ensure metric exists
    if metric_priority not in df.columns:
        st.dataframe(df, hide_index=True, use_container_width=True)
        return

    # Stable ordering: winner → loser, preserve original order for ties
    df["_orig_order"] = range(len(df))
    df = (
        df.sort_values(
            by=[metric_priority, "_orig_order"],
            ascending=[True, True],
            kind="mergesort",
        )
        .drop(columns="_orig_order")
    )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# FORECAST RUN vs RUN COMPARISON (OPTION A)
# --------------------------------------------------

def render_run_comparison():
    history = st.session_state.get("forecast_history", [])
    if len(history) < 2:
        return

    st.markdown("## Forecast Comparison (Run vs Run)")

    # --------------------------------------------------
    # Executive-safe run labels
    # --------------------------------------------------
    labels = []
    for h in history:
        run_id = h.get("run_id", "unknown")
        strategy = h.get("strategy", st.session_state.get("model_selection_strategy", "unknown"))
        winner = h.get("model") or h.get("winner") or "Unknown"

        labels.append(
            f"{run_id} — Strategy: {strategy.capitalize()} | Winner: {winner}"
        )

    col1, col2 = st.columns(2)

    with col1:
        run_a_label = st.selectbox("Run A", labels, index=len(labels) - 2)

    with col2:
        run_b_label = st.selectbox("Run B", labels, index=len(labels) - 1)

    if run_a_label == run_b_label:
        st.info("Select two different runs to compare.")
        return

    ra = history[labels.index(run_a_label)]
    rb = history[labels.index(run_b_label)]

    fa = ra.get("forecast_df")
    fb = rb.get("forecast_df")

    if fa is None or fb is None:
        return

    merged = fa.merge(
        fb,
        on="date",
        how="inner",
        suffixes=("_A", "_B"),
    )

    # Support both forecast naming conventions safely
    if "forecast_A" in merged.columns and "forecast_B" in merged.columns:
        a_col, b_col = "forecast_A", "forecast_B"
    elif "value_A" in merged.columns and "value_B" in merged.columns:
        a_col, b_col = "value_A", "value_B"
    else:
        return

    merged["delta"] = merged[b_col] - merged[a_col]

    st.dataframe(
        merged[["date", a_col, b_col, "delta"]],
        hide_index=True,
        use_container_width=True,
    )
