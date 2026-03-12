# 🔒 LOCKED FILE (TAB MODULE ONLY)
# ==================================================
# FILE: forecastiq/tabs/forecast_results.py
# ROLE: FORECAST RESULTS & AUDIT TABLE (EXECUTIVE-GRADE)
# STATUS: CANONICAL / BENCHMARK-READY
#
# PURPOSE:
# - Provide a numeric, auditable forecast output surface
# - Expose Actual vs Forecast with confidence intervals
# - Serve as the single source of truth for benchmarking
# - Support executive review, audit, and export validation
#
# GOVERNANCE:
# - READ-ONLY
# - NO forecasting logic
# - NO state mutation
# - NO sidebar controls
# - MUST reflect engine outputs exactly
# ==================================================

import streamlit as st
from features.home.forecast_viz import render_baseline_forecast


def render_forecast_results():
    st.header("Forecast Results")
    render_baseline_forecast()
