# ==================================================
# FILE: forecastiq/sidebar/data_setup.py
# ROLE: DATA SETUP SIDEBAR CONTROLS
# STATUS: CONTROL-PLANE SAFE
#
# PURPOSE:
# - Allow user to define data frequency override
# - Define forecast horizon unit + value
#
# GOVERNANCE:
# - Reads/writes ONLY st.session_state
# - No forecasting logic
# - No data mutation
# - No side effects
# ==================================================

import streamlit as st


def render_data_setup_sidebar():
    st.sidebar.subheader("Data Setup")

    # -------------------------------
    # DATA FREQUENCY
    # -------------------------------
    st.sidebar.markdown("**Data frequency**")

    freq_choice = st.sidebar.radio(
        label="",
        options=["Auto", "Monthly", "Weekly"],
        index=["Auto", "Monthly", "Weekly"].index(
            "Auto" if st.session_state.data_frequency_override is None
            else st.session_state.data_frequency_override
        ),
        help="Auto uses inferred frequency. Override only if inference is incorrect."
    )

    if freq_choice == "Auto":
        st.session_state.data_frequency_override = None
    else:
        st.session_state.data_frequency_override = freq_choice


    # -------------------------------
    # FORECAST HORIZON UNIT
    # -------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Forecast horizon unit**")

    st.session_state.forecast_horizon_unit = st.sidebar.radio(
        label="",
        options=["Months", "Weeks"],
        index=["Months", "Weeks"].index(st.session_state.forecast_horizon_unit)
    )


    # -------------------------------
    # FORECAST HORIZON VALUE
    # -------------------------------
    st.sidebar.markdown("**Forecast horizon**")

    st.session_state.forecast_horizon_value = st.sidebar.number_input(
        label="",
        min_value=1,
        max_value=60,
        value=int(st.session_state.forecast_horizon_value),
        step=1,
        help="Number of periods to forecast into the future."
    )
