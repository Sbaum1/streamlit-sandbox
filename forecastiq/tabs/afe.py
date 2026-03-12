# ==================================================
# FILE: forecastiq/tabs/afe.py
# ROLE: ADVANCED FORECAST EXECUTION (AFE) — TAB SHELL
# STATUS: ADDITIVE / GOVERNED / UI-ONLY
#
# PURPOSE:
# - Provide the AFE tab entry point
# - Display execution status, results summary, and diagnostics
# - Act as the executive-facing surface for AFE outputs
#
# GOVERNANCE:
# - NO forecasting logic
# - NO execution logic
# - NO session mutation
# - READ-ONLY consumption of AFE results
# - Safe to add without affecting existing tabs
# ==================================================

import streamlit as st


def render_afe():
    """
    Advanced Forecast Execution (AFE) tab.

    This tab is READ-ONLY and renders outputs produced by the
    AFE orchestration engine. It does not trigger execution.
    """

    st.header("Advanced Forecast Execution (AFE)")
    st.caption(
        "Multi-model forecast execution, diagnostics, and consensus intelligence "
        "for executive decision-making."
    )

    # --------------------------------------------------
    # EXECUTION STATUS
    # --------------------------------------------------
    st.subheader("Execution Status")

    afe_results = st.session_state.get("afe_results")

    if afe_results is None:
        st.info(
            "No AFE execution has been run yet. "
            "AFE results will appear here once execution is completed."
        )
        return

    st.success(f"AFE Execution Complete — {len(afe_results)} models evaluated")

    # --------------------------------------------------
    # MODEL SUMMARY TABLE
    # --------------------------------------------------
    st.subheader("Model Execution Summary")

    summary_rows = []

    for result in afe_results:
        summary_rows.append({
            "Model": result.metadata.model_id,
            "Mode": result.metadata.execution_mode,
            "Forecast Generated": "Yes" if result.forecast is not None else "—",
            "Structural Output": "Yes" if result.structure is not None else "—",
            "Limitations": result.limitations or "None"
        })

    st.dataframe(
        summary_rows,
        use_container_width=True
    )

    # --------------------------------------------------
    # INDIVIDUAL MODEL DETAILS
    # --------------------------------------------------
    st.subheader("Model-Level Details")

    for result in afe_results:
        with st.expander(f"{result.metadata.model_id} — Details"):

            st.markdown("**Execution Metadata**")
            st.json({
                "execution_mode": result.metadata.execution_mode,
                "dataset_hash": result.metadata.dataset_hash,
                "executed_at": result.metadata.executed_at,
                "force_executed": result.is_force_executed,
            })

            if result.forecast is not None:
                st.markdown("**Forecast Output**")
                st.write({
                    "horizon": result.forecast.horizon,
                    "point_forecast": result.forecast.point_forecast,
                })

                if result.forecast.intervals is not None:
                    st.markdown("**Forecast Intervals**")
                    st.write({
                        "base": result.forecast.intervals.base,
                        "upside": result.forecast.intervals.upside,
                        "downside": result.forecast.intervals.downside,
                    })

            if result.structure is not None:
                st.markdown("**Structural / Diagnostic Output**")
                st.write(result.structure)

            if result.limitations:
                st.markdown("**Limitations / Notes**")
                st.warning(result.limitations)

    # --------------------------------------------------
    # GOVERNANCE NOTICE
    # --------------------------------------------------
    st.divider()
    st.caption(
        "AFE outputs are deterministic, auditable, and non-prescriptive. "
        "No model ranking, weighting, or optimization is performed within AFE."
    )
