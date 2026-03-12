# ==================================================
# FILE: sentinel_engine/__init__.py
# VERSION: 2.0.0
# ROLE: PUBLIC INTERFACE — SENTINEL ENGINE
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
#
# GOVERNANCE:
# - This is the ONLY file the Sentinel app imports from
# - No internal submodule imports in the app — ever
# - All public symbols are explicitly declared here
# - Internal implementation details are not exposed
# - Adding a new public symbol requires Architect approval
# - Version is the single source of truth from contracts.py
#
# USAGE (in Sentinel app or any future app):
#
#   from sentinel_engine import (
#       run_all_models,
#       get_model_registry,
#       ForecastResult,
#       certify,
#       verify_certificates,
#       save_report,
#       apply_stress,
#       ENGINE_VERSION,
#   )
#
# ==================================================

from .contracts import (
    ForecastResult,
    ENGINE_VERSION,
)

from .registry import (
    get_model_registry,
    get_ensemble_members,
    get_production_models,
    get_diagnostic_models,
)

from .runner import (
    run_all_models,
    apply_stress,
)

from .certifier import (
    certify,
    verify_certificates,
    generate_certificates,
    save_report,
    hash_forecast,
    hash_dataframe,
    CertificationReport,
    ModelCertResult,
)

from .ensemble import run_primary_ensemble


# --------------------------------------------------
# PUBLIC API DECLARATION
# --------------------------------------------------
# Explicit __all__ prevents accidental exposure of
# internal symbols via wildcard imports.
# --------------------------------------------------

__all__ = [

    # ── Core contract ────────────────────────────
    "ForecastResult",
    "ENGINE_VERSION",

    # ── Registry ─────────────────────────────────
    "get_model_registry",
    "get_ensemble_members",
    "get_production_models",
    "get_diagnostic_models",

    # ── Execution ─────────────────────────────────
    "run_all_models",
    "apply_stress",
    "run_primary_ensemble",

    # ── Certification ─────────────────────────────
    "certify",
    "verify_certificates",
    "generate_certificates",
    "save_report",
    "hash_forecast",
    "hash_dataframe",
    "CertificationReport",
    "ModelCertResult",

]


# --------------------------------------------------
# ENGINE IDENTITY
# --------------------------------------------------

__version__ = ENGINE_VERSION
__author__  = "Sentinel Engine"