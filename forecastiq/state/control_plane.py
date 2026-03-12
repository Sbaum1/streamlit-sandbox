# 🔒 LOCKED FILE
# ==================================================
# FILE: forecastiq/state/control_plane.py
# ROLE: AUTHORITATIVE CONTROL PLANE STATE INITIALIZER
# STATUS: GLOBAL-LOCKED / EXECUTIVE-GRADE
#
# PURPOSE:
# - Initialize all sidebar / control-plane session state keys
# - Enforce single source of truth
# - Prevent silent defaults, drift, or shadow state
#
# GOVERNANCE:
# - NO UI
# - NO MODEL LOGIC
# - NO DATA MUTATION
# - IDEMPOTENT (safe on rerun)
# ==================================================

import streamlit as st


def initialize_control_plane_state():
    """
    Initialize all ForecastIQ control-plane session state keys.
    This function must be called once during app bootstrap
    (e.g., app.py or sidebar initialization).

    It is SAFE to call multiple times.
    """

    # --------------------------------------------------
    # DATA SETUP
    # --------------------------------------------------
    if "data_frequency_override" not in st.session_state:
        # None = trust inferred frequency
        st.session_state.data_frequency_override = None

    if "forecast_horizon_unit" not in st.session_state:
        # Days | Weeks | Months
        st.session_state.forecast_horizon_unit = "Months"

    if "forecast_horizon_value" not in st.session_state:
        st.session_state.forecast_horizon_value = 12


    # --------------------------------------------------
    # MODEL STRATEGY (CANONICAL)
    # --------------------------------------------------
    if "model_selection_strategy" not in st.session_state:
        # auto | prophet | ets | blend | manual
        st.session_state.model_selection_strategy = "auto"

    if "manual_model_choice" not in st.session_state:
        # Used ONLY when model_selection_strategy == "manual"
        # prophet | ets | (future: arima, sarimax, etc.)
        st.session_state.manual_model_choice = None


    # --------------------------------------------------
    # PROPHET ADVANCED SETTINGS
    # --------------------------------------------------
    if "prophet_growth" not in st.session_state:
        # auto | linear | logistic
        st.session_state.prophet_growth = "auto"

    if "prophet_seasonality_mode" not in st.session_state:
        # additive | multiplicative
        st.session_state.prophet_seasonality_mode = "additive"

    if "prophet_changepoint_prior" not in st.session_state:
        # Sensitivity to trend changes
        st.session_state.prophet_changepoint_prior = 0.10

    if "prophet_yearly" not in st.session_state:
        st.session_state.prophet_yearly = True

    if "prophet_weekly" not in st.session_state:
        st.session_state.prophet_weekly = True


    # --------------------------------------------------
    # OUTLIER HANDLING
    # --------------------------------------------------
    if "enable_outlier_cleaning" not in st.session_state:
        st.session_state.enable_outlier_cleaning = False

    if "outlier_z_threshold" not in st.session_state:
        st.session_state.outlier_z_threshold = 3.0


    # --------------------------------------------------
    # SCENARIO SIMULATION
    # --------------------------------------------------
    if "scenario_trend_adjustment" not in st.session_state:
        # Range typically [-0.5, +0.5]
        st.session_state.scenario_trend_adjustment = 0.0
