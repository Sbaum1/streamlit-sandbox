# FILE: execution/run_manager.py
# ROLE: GOVERNED FORECAST EXECUTION CONTROL
# STATUS: CANONICAL
# ==================================================

import streamlit as st

from state.governance import can_run_forecast, lock_forecast_execution
from analysis.forecast_runner import run_all_models


def render_run_controls():
    """
    Executive forecast execution control.

    Governance:
    - Dataset must be committed
    - Forecast may only run once per session
    - All models executed through canonical runner
    """

    st.markdown("---")
    st.subheader("Forecast Execution")

    # ----------------------------------
    # HARD LOCK CHECK
    # ----------------------------------
    if st.session_state.get("forecast_completed", False):
        st.info(
            "Forecast has already been executed for this session. "
            "Reset is required to run again."
        )
        return

    # ----------------------------------
    # DATASET REQUIREMENT
    # ----------------------------------
    if not st.session_state.get("dataset_committed", False):
        st.warning("Dataset must be committed before running a forecast.")
        return

    # ----------------------------------
    # GOVERNANCE CHECK
    # ----------------------------------
    if not can_run_forecast():
        st.warning("Forecast execution is currently locked.")
        return

    # ----------------------------------
    # EXECUTION
    # ----------------------------------
    if st.button("Run Forecast"):
        results = run_all_models(
            df=st.session_state.dataset_df,
            horizon=st.session_state.forecast_horizon,
            confidence=st.session_state.confidence_level,
        )

        st.session_state.forecast_results = results
        st.session_state.forecast_completed = True

        lock_forecast_execution()
        st.success("Forecast executed and locked.")

