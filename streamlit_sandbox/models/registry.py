# ==================================================
# FILE: streamlit_sandbox/models/registry.py
# ROLE: AUTHORITATIVE MODEL REGISTRY
# STATUS: PACKAGE-RELATIVE / EXECUTIVE DEFAULT ENABLED
# ==================================================
#
# GOVERNANCE:
# - Explicit imports only
# - Stable deterministic order
# - No dynamic imports
# - No conditional gating
# - No ranking logic implied
# - Failures handled upstream
# - Package-relative imports only
# ==================================================

from typing import List, Dict

# --------------------------------------------------
# MODEL RUNNERS (PACKAGE-RELATIVE IMPORTS)
# --------------------------------------------------

from .naive import run_naive
from .ets import run_ets
from .arima import run_arima
from .sarima import run_sarima
from .sarimax import run_sarimax
from .theta import run_theta
from .stl_ets import run_stl_ets
from .tbats import run_tbats
from .prophet_model import run_prophet
from .bsts import run_bsts
from .x13 import run_x13

# --------------------------------------------------
# PRIMARY ENSEMBLE (CERTIFIED EXECUTIVE DEFAULT)
# --------------------------------------------------

from streamlit_sandbox.models.primary_ensemble import run_primary_ensemble


# --------------------------------------------------
# AUTHORITATIVE MODEL REGISTRY
# --------------------------------------------------

def get_model_registry() -> List[Dict]:
    """
    Canonical forecasting model registry.

    Enterprise Governance:
    - Deterministic ordering
    - Primary Ensemble listed first (Executive Default)
    - No ranking implied
    - No silent fallbacks
    - No environment-dependent imports
    """

    return [
        # --------------------------------------------------
        # EXECUTIVE DEFAULT (PRIMARY ENSEMBLE)
        # --------------------------------------------------
        {"name": "Primary Ensemble", "runner": run_primary_ensemble},

        # --------------------------------------------------
        # INDIVIDUAL MODELS (AUDITABLE / SELECTABLE)
        # --------------------------------------------------
        {"name": "Naive", "runner": run_naive},
        {"name": "ETS", "runner": run_ets},
        {"name": "ARIMA", "runner": run_arima},
        {"name": "SARIMA", "runner": run_sarima},
        {"name": "SARIMAX", "runner": run_sarimax},
        {"name": "Theta", "runner": run_theta},
        {"name": "STL+ETS", "runner": run_stl_ets},
        {"name": "TBATS", "runner": run_tbats},
        {"name": "Prophet", "runner": run_prophet},
        {"name": "BSTS", "runner": run_bsts},
        {"name": "X-13", "runner": run_x13},
    ]