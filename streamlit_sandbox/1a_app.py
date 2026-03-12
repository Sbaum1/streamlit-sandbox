import streamlit as st

# ==================================================
# GOVERNED APPLICATION BOOTSTRAP
# ==================================================
from state.session_state import init_state
from models.registry import get_model_registry
import home


# --------------------------------------------------
# STREAMLIT CONFIG (EXECUTIVE GRADE)
# --------------------------------------------------
st.set_page_config(
    page_title="Executive Forecasting Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# SESSION STATE INITIALIZATION (ONCE)
# --------------------------------------------------
init_state()


# --------------------------------------------------
# EXECUTIVE DEFAULT MODEL LOCK
# --------------------------------------------------

registry = get_model_registry()

available_models = [m["name"] for m in registry]

EXECUTIVE_DEFAULT = "Primary Ensemble"

if EXECUTIVE_DEFAULT not in available_models:
    raise RuntimeError("Primary Ensemble missing from registry. Governance violation.")

# Only set default if user has not selected a model yet
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = EXECUTIVE_DEFAULT


# --------------------------------------------------
# ROUTE TO EXECUTIVE HOME SURFACE
# --------------------------------------------------
home.render()
