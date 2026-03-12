import streamlit as st
import uuid


def init_state():
    """
    Initialize all governed session state keys.
    This function is idempotent and safe across Streamlit reruns.
    """

    defaults = {
        # --------------------------------------------------
        # DATASET GOVERNANCE
        # --------------------------------------------------
        "dataset_committed": False,
        "dataset_df": None,
        "dataset_fingerprint": None,

        # --------------------------------------------------
        # ANALYSIS STATE
        # --------------------------------------------------
        "analysis_completed": False,

        # --------------------------------------------------
        # FORECAST EXECUTION GOVERNANCE
        # --------------------------------------------------
        "forecast_locked": False,     # Prevents re-execution
        "forecast_executed": False,   # Tracks if forecast has run
        "run_id": None,               # Unique run identifier
        "run_timestamp": None,        # Populated at execution time

        # --------------------------------------------------
        # EXECUTIVE SIDEBAR â€” CORE CONTROLS
        # --------------------------------------------------
        "forecast_horizon": 12,
        "backtest_window": 24,
        "confidence_level": 0.9,
        "optimization_mode": "Auto",  # Auto | Manual

        # --------------------------------------------------
        # EXECUTIVE SIDEBAR â€” MACRO VARIABLES
        # --------------------------------------------------
        "macro_factors": {
            "interest_rates": "Medium",
            "credit_availability": "Medium",
            "unemployment": "Medium",
            "housing_starts": "Medium",
            "consumer_confidence": "Medium",
            "inflation": "Medium",
            "construction_spend": "Medium",
        },

        # --------------------------------------------------
        # EXECUTIVE SIDEBAR â€” SCENARIO PRESETS
        # --------------------------------------------------
        "recession_profile": "None",  # None | Light | Moderate | Severe
    }

    # --------------------------------------------------
    # APPLY DEFAULTS (DO NOT OVERWRITE EXISTING STATE)
    # --------------------------------------------------
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

