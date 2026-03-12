# 🔒 LOCKED FILE (READ-ONLY EXECUTIVE SUMMARY)
# ==================================================
# FILE: features/home/scenario_impact_summary.py
# ROLE: SCENARIO IMPACT — EXECUTIVE SUMMARY PANEL
# STATUS: ADDITIVE / GOVERNANCE-COMPLIANT
#
# PURPOSE:
# - Summarize scenario impact vs baseline
# - Translate math into executive insight
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


def render_scenario_impact_summary():
    """
    Render an executive summary of scenario impact vs baseline forecast.

    Requires (read-only):
      - st.session_state.latest_forecast_df        (baseline)
      - st.session_state.scenario_forecast_df      (scenario)
      - st.session_state.active_scenario_audit     (metadata)
    """

    baseline = st.session_state.get("latest_forecast_df")
    scenario = st.session_state.get("scenario_forecast_df")
    audit = st.session_state.get("active_scenario_audit")

    # --------------------------------------------------
    # GUARDRAILS
    # --------------------------------------------------
    if baseline is None or scenario is None or audit is None:
        return

    if not isinstance(baseline, pd.DataFrame) or not isinstance(scenario, pd.DataFrame):
        return

    required_cols = {"date", "forecast"}
    if not required_cols.issubset(baseline.columns):
        return
    if not required_cols.issubset(scenario.columns):
        return

    # --------------------------------------------------
    # ALIGN DATA
    # --------------------------------------------------
    b = baseline[["date", "forecast"]].rename(columns={"forecast": "baseline"})
    s = scenario[["date", "forecast"]].rename(columns={"forecast": "scenario"})

    merged = b.merge(s, on="date", how="inner")

    if merged.empty:
        return

    # --------------------------------------------------
    # IMPACT METRICS
    # --------------------------------------------------
    merged["delta"] = merged["scenario"] - merged["baseline"]

    cumulative_baseline = merged["baseline"].sum()
    cumulative_scenario = merged["scenario"].sum()
    cumulative_delta = cumulative_scenario - cumulative_baseline

    worst_delta = merged["delta"].min()
    best_delta = merged["delta"].max()

    pct_impact = (
        cumulative_delta / cumulative_baseline
        if cumulative_baseline != 0
        else np.nan
    )

    # --------------------------------------------------
    # EXECUTIVE NARRATIVE
    # --------------------------------------------------
    scenario_name = audit.get("scenario_name", "Scenario")
    description = audit.get("scenario_description", "")

    direction = "negative" if cumulative_delta < 0 else "positive"

    if pd.notna(pct_impact):
        magnitude_pct = abs(pct_impact) * 100
        narrative = (
            f"**{scenario_name}** introduces a **{direction} impact** over the "
            f"forecast horizon, changing cumulative expectations by "
            f"**{magnitude_pct:.1f}%** relative to baseline."
        )
    else:
        narrative = (
            f"**{scenario_name}** impact summary could not be computed "
            f"due to insufficient baseline magnitude."
        )

    # --------------------------------------------------
    # PRESENTATION
    # --------------------------------------------------
    st.markdown("### Scenario Impact Summary")

    if description:
        st.caption(description)

    st.markdown(narrative)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Cumulative Δ vs Baseline",
            f"{cumulative_delta:,.0f}",
        )

    with c2:
        st.metric(
            "Worst Point Impact",
            f"{worst_delta:,.0f}",
        )

    with c3:
        st.metric(
            "Best Point Impact",
            f"{best_delta:,.0f}",
        )
