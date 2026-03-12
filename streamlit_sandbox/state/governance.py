# FILE: state/governance.py
# ROLE: EXECUTION & DATA GOVERNANCE (CANONICAL)
# STATUS: LOCKED
# ==================================================

import streamlit as st
import uuid
from datetime import datetime, timezone


# ==================================================
# DATASET COMMIT GOVERNANCE
# ==================================================

def lock_after_commit():
    """
    Locks the dataset after explicit commitment.
    Does NOT execute a forecast.
    """
    st.session_state.dataset_committed = True
    st.session_state.analysis_completed = True


# ==================================================
# FORECAST EXECUTION GOVERNANCE
# ==================================================

def can_run_forecast() -> bool:
    """
    Returns True if a forecast execution is allowed.

    Governance:
    - Dataset must be committed
    - Forecast must not have already completed
    - Forecast must not be explicitly locked
    """
    return (
        st.session_state.get("dataset_committed", False)
        and not st.session_state.get("forecast_completed", False)
        and not st.session_state.get("forecast_locked", False)
    )


def lock_forecast_execution():
    """
    Locks forecast execution after a single run.
    Generates immutable execution metadata.
    """
    st.session_state.forecast_completed = True
    st.session_state.forecast_locked = True
    st.session_state.run_id = str(uuid.uuid4())
    st.session_state.run_timestamp = datetime.now(timezone.utc)


# ==================================================
# RESET GOVERNANCE
# ==================================================

def reset_all():
    """
    Hard reset of the entire Streamlit session state.
    Required before any new dataset commit or forecast run.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]

