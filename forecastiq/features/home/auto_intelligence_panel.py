# ==================================================
# FILE: features/home/auto_intelligence_panel.py
# ROLE: AUTO-INTELLIGENCE SUMMARY & RATIONALE (BODY-ONLY)
# STATUS: EXECUTIVE-GRADE / LOCKED
#
# GOVERNANCE:
# - NO st.sidebar calls allowed in this file
# - This file renders BODY content ONLY
# - Toggle is handled elsewhere (sidebar controller)
# ==================================================

import streamlit as st
import pandas as pd


def render_auto_intelligence_summary():
    """
    Body-only renderer.
    Assumes the Auto-Intelligence toggle has already been set in session_state
    by a sidebar controller.
    """

    if not st.session_state.get("auto_intelligence", False):
        return

    df = st.session_state.get("committed_df")
    freq = st.session_state.get("data_frequency")
    horizon = st.session_state.get("forecast_horizon")
    metric = st.session_state.get("metric_priority")
    confidence = st.session_state.get("confidence_level")

    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    st.markdown("## Auto-Intelligence Rationale")

    st.markdown(
        f"""
**Why these settings were selected**

• **Data length:** {len(df)} observations  
• **Detected frequency:** {freq}  
• **Forecast horizon:** {horizon} periods — aligned to executive planning cycles  
• **Metric priority:** {metric} — optimized for reliability at this data scale  
• **Confidence level:** {confidence}% — balanced for accuracy vs. risk
"""
    )

    st.caption(
        "Auto-Intelligence provides guidance only. Executives retain full control over all settings."
    )
