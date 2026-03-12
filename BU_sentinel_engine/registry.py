# ==================================================
# FILE: sentinel_engine/registry.py
# VERSION: 2.0.0
# ROLE: AUTHORITATIVE MODEL REGISTRY
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - Explicit imports only
# - Stable deterministic order
# - No dynamic imports
# - No conditional gating
# - No ranking logic
# - Failures handled upstream in runner.py
# - Package-relative imports only
# - X-13 registered as diagnostic_only
# ==================================================

from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------
# PRODUCTION MODEL RUNNERS
# --------------------------------------------------

from .models.naive       import run_naive
from .models.ets         import run_ets
from .models.arima       import run_arima
from .models.sarima      import run_sarima
from .models.sarimax     import run_sarimax
from .models.theta       import run_theta
from .models.stl_ets     import run_stl_ets
from .models.tbats       import run_tbats
from .models.prophet     import run_prophet
from .models.bsts        import run_bsts
from .models.x13         import run_x13

# --------------------------------------------------
# PRIMARY ENSEMBLE
# --------------------------------------------------

from .ensemble import run_primary_ensemble


# --------------------------------------------------
# AUTHORITATIVE MODEL REGISTRY
# --------------------------------------------------

def get_model_registry() -> List[Dict[str, Any]]:
    """
    Canonical forecasting model registry for Sentinel Engine v2.0.0.

    Governance:
    - Deterministic ordering — Primary Ensemble listed first
    - Production models listed in stable alphabetical order after ensemble
    - Naive listed last — baseline reference only, not an ensemble member
    - X-13 registered as diagnostic_only — runs but excluded from scoring
      and ensemble aggregation
    - No silent fallbacks
    - No environment-dependent imports
    - No ranking implied by order

    Registry Entry Schema:
        name            : str  — display name, used as result key
        runner          : callable(df, horizon, confidence_level) -> ForecastResult
        diagnostic_only : bool — if True, runner executes but metrics and
                                 readiness scoring are skipped; result is
                                 excluded from ensemble aggregation
        ensemble_member : bool — if True, runner output is eligible for
                                 inclusion in Primary Ensemble aggregation
    """

    return [

        # --------------------------------------------------
        # EXECUTIVE DEFAULT (PRIMARY ENSEMBLE)
        # --------------------------------------------------
        {
            "name":             "Primary Ensemble",
            "runner":           run_primary_ensemble,
            "diagnostic_only":  False,
            "ensemble_member":  False,      # Ensemble is the output, not an input
        },

        # --------------------------------------------------
        # PRODUCTION MODELS — ENSEMBLE MEMBERS (7)
        # --------------------------------------------------
        {
            "name":             "BSTS",
            "runner":           run_bsts,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },
        {
            "name":             "ETS",
            "runner":           run_ets,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },
        {
            "name":             "Prophet",
            "runner":           run_prophet,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },
        {
            "name":             "SARIMA",
            "runner":           run_sarima,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },
        {
            "name":             "STL+ETS",
            "runner":           run_stl_ets,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },
        {
            "name":             "TBATS",
            "runner":           run_tbats,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },
        {
            "name":             "Theta",
            "runner":           run_theta,
            "diagnostic_only":  False,
            "ensemble_member":  True,
        },

        # --------------------------------------------------
        # PRODUCTION MODELS — INDIVIDUAL ONLY (NOT ENSEMBLE MEMBERS)
        # --------------------------------------------------
        {
            "name":             "ARIMA",
            "runner":           run_arima,
            "diagnostic_only":  False,
            "ensemble_member":  False,      # Covered by SARIMA in ensemble
        },
        {
            "name":             "SARIMAX",
            "runner":           run_sarimax,
            "diagnostic_only":  False,
            "ensemble_member":  False,      # Requires exogenous data — selectable only
        },

        # --------------------------------------------------
        # BASELINE REFERENCE (NOT AN ENSEMBLE MEMBER)
        # --------------------------------------------------
        {
            "name":             "Naive",
            "runner":           run_naive,
            "diagnostic_only":  False,
            "ensemble_member":  False,      # Baseline only — used for MASE scaling
        },

        # --------------------------------------------------
        # DIAGNOSTIC ONLY (EXCLUDED FROM SCORING + ENSEMBLE)
        # --------------------------------------------------
        {
            "name":             "X-13",
            "runner":           run_x13,
            "diagnostic_only":  True,       # Confirmed empty output — diagnostic only
            "ensemble_member":  False,
        },

    ]


# --------------------------------------------------
# REGISTRY ACCESSORS
# --------------------------------------------------

def get_ensemble_members() -> List[Dict[str, Any]]:
    """
    Returns only the registry entries eligible for ensemble aggregation.
    Used by ensemble.py to retrieve member runners without coupling to
    the full registry.
    """
    return [
        entry for entry in get_model_registry()
        if entry.get("ensemble_member", False)
    ]


def get_production_models() -> List[Dict[str, Any]]:
    """
    Returns all non-diagnostic registry entries.
    Used by runner.py for full execution pass.
    """
    return [
        entry for entry in get_model_registry()
        if not entry.get("diagnostic_only", False)
    ]


def get_diagnostic_models() -> List[Dict[str, Any]]:
    """
    Returns only diagnostic_only registry entries.
    """
    return [
        entry for entry in get_model_registry()
        if entry.get("diagnostic_only", False)
    ]