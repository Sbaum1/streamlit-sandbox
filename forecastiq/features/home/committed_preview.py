# ==================================================
# FILE: forecastiq/features/home/committed_preview.py
# ROLE: COMMITTED DATA PREVIEW + DATA PROFILE (EXTRACTED)
# STATUS: SAFE FEATURE EXTRACTION — NO SIDE EFFECTS
# ==================================================

import streamlit as st
import pandas as pd


def render_committed_preview():
    """
    Renders the committed data preview and associated data profile metadata.

    SAFETY CONTRACT:
    - Reads ONLY from st.session_state
    - Does NOT mutate session state
    - Does NOT compute forecasts
    - Renders exactly:
        1) One preview table
        2) Metadata directly beneath it
    """

    df = st.session_state.get("committed_df")
    freq = st.session_state.get("data_frequency")
    details = st.session_state.get("freq_inference_details", {})

    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    # -------------------------------
    # COMMITTED DATA PREVIEW
    # -------------------------------
    st.markdown("### Committed Data Preview")

    st.dataframe(
        df,
        use_container_width=True,
    )

    # -------------------------------
    # DATA PROFILE / METADATA
    # -------------------------------
    rows = len(df)
    frequency = freq or "Unknown"
    confidence = details.get("confidence", 0.0)
    info = details.get("details", "")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", rows)
    c2.metric("Detected Frequency", frequency)
    c3.metric("Frequency Confidence", f"{confidence:.0%}")

    if info:
        st.caption(info)
