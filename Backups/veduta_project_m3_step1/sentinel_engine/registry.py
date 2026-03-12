# ==================================================
# FILE: sentinel_engine/registry.py
# VERSION: 3.0.0
# ROLE: AUTHORITATIVE MODEL REGISTRY
# ENGINE: Sentinel Engine v2.0.0
# UPDATED: PHASE 3C — 7 NEW MODELS ADDED
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
#
# PHASE 3C ADDITIONS (7 new models):
# - HW Damped    : essentials tier, ensemble member
# - Croston/SBA  : pro tier, intermittent demand routing
# - DHR          : pro tier, ensemble member
# - NNETAR       : enterprise tier, ensemble member
# - LightGBM     : pro tier, ensemble member
# - VAR          : enterprise tier, multi-series only
# - GARCH        : enterprise tier, volatility CI modifier
#
# TIER TAGS (min_tier):
#   essentials — available in all three platform tiers
#   pro        — available in Pro and Enterprise
#   enterprise — available in Enterprise only
#
# ROUTING NOTES:
#   Croston/SBA : route series with >30% zero periods here
#   VAR         : requires multi-series df (2+ numeric columns)
#   SARIMAX     : requires exogenous columns in df
#   GARCH       : volatility_forecast in metadata for Phase 3D CI scaling
# ==================================================

from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------
# PRODUCTION MODEL RUNNERS — PHASE 1-3B (EXISTING)
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
# PRODUCTION MODEL RUNNERS — PHASE 3C (NEW)
# --------------------------------------------------

from .models.hw_damped       import run_hw_damped
from .models.croston         import run_croston
from .models.dhr             import run_dhr
from .models.nnetar          import run_nnetar
from .models.lightgbm_model  import run_lightgbm
from .models.var_model       import run_var
from .models.garch_model     import run_garch

# --------------------------------------------------
# PRIMARY ENSEMBLE
# --------------------------------------------------

from .ensemble import run_primary_ensemble


# --------------------------------------------------
# AUTHORITATIVE MODEL REGISTRY
# --------------------------------------------------

def get_model_registry() -> List[Dict[str, Any]]:
    """
    Canonical forecasting model registry for Sentinel Engine v2.0.0 Phase 3C.

    Governance:
    - Deterministic ordering — Primary Ensemble listed first
    - Existing models listed in stable alphabetical order
    - Phase 3C models listed after existing, grouped by tier
    - Naive listed last among production models — baseline reference
    - X-13 registered as diagnostic_only
    - No silent fallbacks
    - No environment-dependent imports
    - No ranking implied by order

    Registry Entry Schema:
        name            : str      — display name, used as result key
        runner          : callable(df, horizon, confidence_level) -> ForecastResult
        diagnostic_only : bool     — if True, excluded from scoring and ensemble
        ensemble_member : bool     — if True, eligible for Primary Ensemble
        min_tier        : str      — minimum platform tier: essentials | pro | enterprise
        routing_note    : str|None — special input requirements or routing conditions
    """

    return [

        # --------------------------------------------------
        # EXECUTIVE DEFAULT (PRIMARY ENSEMBLE)
        # --------------------------------------------------
        {
            "name":             "Primary Ensemble",
            "runner":           run_primary_ensemble,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     None,
        },

        # --------------------------------------------------
        # PRODUCTION MODELS — EXISTING ENSEMBLE MEMBERS (7)
        # --------------------------------------------------
        {
            "name":             "BSTS",
            "runner":           run_bsts,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "ETS",
            "runner":           run_ets,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "Prophet",
            "runner":           run_prophet,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "SARIMA",
            "runner":           run_sarima,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "STL+ETS",
            "runner":           run_stl_ets,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "TBATS",
            "runner":           run_tbats,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },
        {
            "name":             "Theta",
            "runner":           run_theta,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     None,
        },

        # --------------------------------------------------
        # PRODUCTION MODELS — EXISTING INDIVIDUAL ONLY
        # --------------------------------------------------
        {
            "name":             "ARIMA",
            "runner":           run_arima,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     None,
        },
        {
            "name":             "SARIMAX",
            "runner":           run_sarimax,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     "Requires exogenous columns in df alongside 'date' and 'value'.",
        },

        # --------------------------------------------------
        # BASELINE REFERENCE
        # --------------------------------------------------
        {
            "name":             "Naive",
            "runner":           run_naive,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "essentials",
            "routing_note":     None,
        },

        # --------------------------------------------------
        # DIAGNOSTIC ONLY
        # --------------------------------------------------
        {
            "name":             "X-13",
            "runner":           run_x13,
            "diagnostic_only":  True,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     None,
        },

        # --------------------------------------------------
        # PHASE 3C — TIER: ESSENTIALS
        # --------------------------------------------------
        {
            "name":             "HW_Damped",
            "runner":           run_hw_damped,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "essentials",
            "routing_note":     "Preferred over ETS on short-horizon volatile series.",
        },

        # --------------------------------------------------
        # PHASE 3C — TIER: PRO
        # --------------------------------------------------
        {
            "name":             "Croston_SBA",
            "runner":           run_croston,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "pro",
            "routing_note":     "Route series with >30% zero periods to this model. "
                                "Auto-excluded from standard ensemble on non-intermittent series.",
        },
        {
            "name":             "DHR",
            "runner":           run_dhr,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Preferred on series with multiple seasonality periods.",
        },
        {
            "name":             "LightGBM",
            "runner":           run_lightgbm,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "pro",
            "routing_note":     "Accepts exogenous columns (CPI, promo flags, pricing) "
                                "in df alongside 'date' and 'value'. Falls back to lag-only "
                                "if future exog values not provided.",
        },

        # --------------------------------------------------
        # PHASE 3C — TIER: ENTERPRISE
        # --------------------------------------------------
        {
            "name":             "NNETAR",
            "runner":           run_nnetar,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "Best on volatile non-linear series. Requires >= 16 obs.",
        },
        {
            "name":             "VAR",
            "runner":           run_var,
            "diagnostic_only":  False,
            "ensemble_member":  False,
            "min_tier":         "enterprise",
            "routing_note":     "Requires df with 2+ numeric series columns. "
                                "Primary series = 'value' column or first numeric column. "
                                "Will raise ValueError on single-series input.",
        },
        {
            "name":             "GARCH",
            "runner":           run_garch,
            "diagnostic_only":  False,
            "ensemble_member":  True,
            "min_tier":         "enterprise",
            "routing_note":     "volatility_forecast in metadata available for "
                                "CI width scaling in Phase 3D ensemble upgrade.",
        },

    ]


# --------------------------------------------------
# REGISTRY ACCESSORS
# --------------------------------------------------

def get_ensemble_members() -> List[Dict[str, Any]]:
    """Returns only ensemble-eligible registry entries."""
    return [
        entry for entry in get_model_registry()
        if entry.get("ensemble_member", False)
    ]


def get_production_models() -> List[Dict[str, Any]]:
    """Returns all non-diagnostic registry entries."""
    return [
        entry for entry in get_model_registry()
        if not entry.get("diagnostic_only", False)
    ]


def get_diagnostic_models() -> List[Dict[str, Any]]:
    """Returns only diagnostic_only registry entries."""
    return [
        entry for entry in get_model_registry()
        if entry.get("diagnostic_only", False)
    ]


def get_models_by_tier(tier: str) -> List[Dict[str, Any]]:
    """
    Returns all registry entries available at a given platform tier.

    Args:
        tier: 'essentials' | 'pro' | 'enterprise'

    Tier hierarchy:
        essentials -> essentials models only
        pro        -> essentials + pro models
        enterprise -> all models
    """
    tier_order = {"essentials": 0, "pro": 1, "enterprise": 2}
    tier_level = tier_order.get(tier, 2)
    return [
        entry for entry in get_model_registry()
        if tier_order.get(entry.get("min_tier", "enterprise"), 2) <= tier_level
    ]


def get_ensemble_members_by_tier(tier: str) -> List[Dict[str, Any]]:
    """
    Returns ensemble-eligible models available at a given tier.
    Used by ensemble.py when tier-aware execution is active.
    """
    return [
        entry for entry in get_models_by_tier(tier)
        if entry.get("ensemble_member", False)
    ]
