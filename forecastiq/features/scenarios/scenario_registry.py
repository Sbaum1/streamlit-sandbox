# 🔒 LOCKED FILE (DECLARATIVE REGISTRY ONLY)
# ==================================================
# FILE: forecastiq/features/scenarios/scenario_registry.py
# ROLE: Approved Scenario Registry (Declarative)
# STATUS: CANONICAL / GOVERNANCE-COMPLIANT
#
# PURPOSE:
# - Central registry of approved scenario definitions
# - Provides auditable, named scenarios for UI selection
# - Encodes executive INTENT and SEVERITY as first-class signals
#
# GOVERNANCE:
# - NO logic
# - NO computation
# - NO engine access
# - NO UI access
# - DATA ONLY
#
# NOTE:
# - Scenarios are interpreted by scenario_adapter.py
# - Verdict posture is interpreted by decision_surface.py
# - This file may later be replaced by JSON / DB without code changes
# ==================================================


# --------------------------------------------------
# BASELINE (IDENTITY — EXECUTIVE NEUTRAL)
# --------------------------------------------------

BASELINE_SCENARIO = {
    "id": "baseline",
    "name": "Baseline",
    "description": "Unmodified engine forecast (source of truth).",
    "intent": "baseline",          # executive posture
    "severity": "none",             # must be none for baseline
    "type": "identity",
    "parameters": {},
}


# --------------------------------------------------
# EXECUTIVE RECESSION SCENARIOS (INTENTIONAL STRESS)
# --------------------------------------------------

RECESSION_LIGHT = {
    "id": "recession_light",
    "name": "Recession (Light)",
    "description": "Executive assumes a mild recession environment.",
    "intent": "recession",          # executive intent
    "severity": "light",             # minimum caution posture
    "type": "shock",
    "parameters": {
        "magnitude_pct": -5.0,
        "recovery_periods": 4,
    },
}

RECESSION_SEVERE = {
    "id": "recession_severe",
    "name": "Recession (Severe)",
    "description": "Executive assumes a severe recession environment.",
    "intent": "recession",
    "severity": "severe",            # minimum defensive posture
    "type": "shock",
    "parameters": {
        "magnitude_pct": -20.0,
        "recovery_periods": 8,
        "shock_factor": 0.80,
        "horizon": 8,
    },
}


# --------------------------------------------------
# NON-RECESSION STRUCTURAL SCENARIOS
# --------------------------------------------------

GROWTH_ACCELERATION = {
    "id": "growth_acceleration",
    "name": "Growth Acceleration",
    "description": "Permanent +6% level shift across forecast horizon.",
    "intent": "baseline",
    "severity": "none",
    "type": "level_shift",
    "parameters": {
        "pct": 6.0,
    },
}

TREND_DECELERATION = {
    "id": "trend_deceleration",
    "name": "Trend Deceleration",
    "description": "Gradual slowdown from 0% to -6% across forecast horizon.",
    "intent": "baseline",
    "severity": "none",
    "type": "trend_change",
    "parameters": {
        "start_pct": 0.0,
        "end_pct": -6.0,
    },
}


# --------------------------------------------------
# REGISTRY (AUTHORITATIVE)
# --------------------------------------------------

SCENARIO_REGISTRY = {
    BASELINE_SCENARIO["id"]: BASELINE_SCENARIO,
    RECESSION_LIGHT["id"]: RECESSION_LIGHT,
    RECESSION_SEVERE["id"]: RECESSION_SEVERE,
    GROWTH_ACCELERATION["id"]: GROWTH_ACCELERATION,
    TREND_DECELERATION["id"]: TREND_DECELERATION,
}


# --------------------------------------------------
# ORDERING (UI-FRIENDLY)
# --------------------------------------------------

SCENARIO_ORDER = [
    "baseline",
    "recession_light",
    "recession_severe",
    "growth_acceleration",
    "trend_deceleration",
]
