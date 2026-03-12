# ==================================================
# FILE: forecastiq/sidebar/model_strategy.py
# ROLE: MODEL STRATEGY SELECTOR
# STATUS: CONTROL-PLANE SAFE / EXECUTIVE-GRADE
#
# PURPOSE:
# - Allow executives/analysts to define how models are selected
# - Strategy is advisory until forecast execution
#
# GOVERNANCE:
# - Reads/writes ONLY st.session_state
# - No forecasting logic
# - No data mutation
# - No side effects
# ==================================================

import streamlit as st


def render_model_strategy_sidebar():
    st.sidebar.subheader("Model Strategy")

    st.sidebar.markdown(
        "Controls how ForecastIQ selects or constrains forecasting models."
    )

    strategy_options = {
        "Automatic (Best Fit)": {
            "key": "auto",
            "desc": "Evaluate all eligible models and select the best performer."
        },
        "Prophet Only": {
            "key": "prophet_only",
            "desc": "Restrict forecasting to Prophet models only."
        },
        "Classical Models Only": {
            "key": "classical_only",
            "desc": "Use ETS / ARIMA-class models only."
        },
        "Manual Selection": {
            "key": "manual",
            "desc": "Allow analyst to explicitly choose the model."
        },
    }

    labels = list(strategy_options.keys())

    selected_label = st.sidebar.radio(
        label="",
        options=labels,
        index=labels.index(
            next(
                k for k, v in strategy_options.items()
                if v["key"] == st.session_state.model_strategy
            )
            if st.session_state.get("model_strategy") is not None
            else 0
        )
    )

    # Persist selection
    st.session_state.model_strategy = strategy_options[selected_label]["key"]

    # Description block
    st.sidebar.caption(strategy_options[selected_label]["desc"])

    # --------------------------------------------------
    # MANUAL MODEL OVERRIDE (VISIBLE ONLY IF ENABLED)
    # --------------------------------------------------
    if st.session_state.model_strategy == "manual":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Manual model selection**")

        model_choices = [
            "Prophet",
            "ETS (Holt-Winters)",
            "ARIMA",
            "SARIMA"
        ]

        st.session_state.manual_model_choice = st.sidebar.selectbox(
            label="Select model",
            options=model_choices,
            index=model_choices.index(
                st.session_state.manual_model_choice
            )
            if st.session_state.get("manual_model_choice") in model_choices
            else 0
        )
