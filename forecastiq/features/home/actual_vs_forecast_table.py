# 🔒 LOCKED FILE (READ-ONLY RENDERER)
# ==================================================
# FILE: features/home/actual_vs_forecast_table.py
# ROLE: ACTUAL VS FORECAST — EXECUTIVE TABLE
# STATUS: ADDITIVE / GOVERNANCE-COMPLIANT
#
# PURPOSE:
# - Present row-level comparison of actuals vs forecast
# - Executive auditability and transparency
#
# GOVERNANCE:
# - READS st.session_state only
# - NO mutation
# - NO forecasting logic
# - NO side effects
# ==================================================

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np


def render_actual_vs_forecast_table():
    """
    Render an executive-grade Actual vs Forecast comparison table.

    Requires (read-only):
      - st.session_state.committed_df
      - st.session_state.latest_forecast_df
      - st.session_state.latest_model_name

    Notes:
      - Uses canonical forecast dataframe produced by engine
      - Displays only rows where actuals exist
      - Confidence intervals shown if present
    """

    actual_df = st.session_state.get("committed_df")
    forecast_df = st.session_state.get("latest_forecast_df")
    model_name = st.session_state.get("latest_model_name")

    # --------------------------------------------------
    # GUARDRAILS
    # --------------------------------------------------
    if actual_df is None or forecast_df is None:
        return

    if not isinstance(actual_df, pd.DataFrame):
        return

    if not isinstance(forecast_df, pd.DataFrame):
        return

    required_actual_cols = {"date", "value"}
    required_forecast_cols = {"date", "actual", "forecast"}

    if not required_actual_cols.issubset(actual_df.columns):
        return

    if not required_forecast_cols.issubset(forecast_df.columns):
        return

    # --------------------------------------------------
    # NORMALIZE ACTUALS
    # --------------------------------------------------
    actuals = (
        actual_df[["date", "value"]]
        .rename(columns={"value": "actual"})
        .copy()
    )

    actuals["date"] = pd.to_datetime(actuals["date"])

    # --------------------------------------------------
    # NORMALIZE FORECAST FRAME
    # --------------------------------------------------
    fc = forecast_df.copy()
    fc["date"] = pd.to_datetime(fc["date"])

    # Use only rows where actuals exist (historical overlap)
    fc = fc[fc["actual"].notna()]

    if fc.empty:
        return

    # --------------------------------------------------
    # JOIN (DATE-ALIGNED, ENGINE-SAFE)
    # --------------------------------------------------
    table = actuals.merge(
        fc[["date", "forecast", "lower", "upper"]],
        on="date",
        how="inner",
    )

    if table.empty:
        return

    # --------------------------------------------------
    # ERROR METRICS
    # --------------------------------------------------
    table["error"] = table["actual"] - table["forecast"]
    table["abs_error"] = table["error"].abs()

    table["pct_error"] = np.where(
        table["actual"] != 0,
        table["error"] / table["actual"],
        np.nan,
    )

    # --------------------------------------------------
    # PRESENTATION
    # --------------------------------------------------
    st.markdown("### Actual vs Forecast (Row-Level)")

    st.caption(
        f"Comparison of committed actuals against the selected forecast model "
        f"({model_name}). Errors are computed where dates overlap."
    )

    st.dataframe(
        table[
            [
                "date",
                "actual",
                "forecast",
                "error",
                "abs_error",
                "pct_error",
                "lower",
                "upper",
            ]
        ],
        use_container_width=True,
    )
