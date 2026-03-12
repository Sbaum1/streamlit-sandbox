# 🔒 LOCKED FILE (SCENARIO ADAPTER — PURE / NO UI)
# ==================================================
# FILE: forecastiq/features/scenarios/scenario_adapter.py
# ROLE: Scenario transformation adapter (Baseline → Scenario)
# STATUS: CANONICAL / INTENT-AWARE / EXECUTIVE-GRADE
#
# PURPOSE:
# - Apply deterministic, auditable scenario transforms
# - Incorporate executive intent as a first-class modifier
#
# GOVERNANCE:
# - NO Streamlit imports
# - NO session_state access
# - NO forecasting / model fitting
# - PURE functions only
# - DOES NOT mutate baseline_df
#
# REQUIRED INPUT CONTRACT (baseline_df):
# - Must contain columns: date, forecast, is_future
# - Optional columns: lower, upper
#
# OUTPUT:
# - Returns (scenario_df, audit)
# - scenario_df preserves schema + adds scenario metadata
# ==================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    name: str
    description: str
    type: str
    parameters: Dict[str, Any]


# ==================================================
# VALIDATION
# ==================================================

_ALLOWED_TYPES = {"identity", "level_shift", "trend_change", "shock"}


def validate_scenario(s: Dict[str, Any]) -> ScenarioSpec:
    if not isinstance(s, dict):
        raise ValueError("Scenario must be a dict.")

    scenario_id = str(s.get("id") or "").strip()
    name = str(s.get("name") or "").strip()
    description = str(s.get("description") or "").strip()
    stype = str(s.get("type") or "").strip()
    params = s.get("parameters") or {}

    if not scenario_id or not name:
        raise ValueError("Scenario missing required id or name.")
    if stype not in _ALLOWED_TYPES:
        raise ValueError(f"Unsupported scenario type: {stype}")
    if not isinstance(params, dict):
        raise ValueError("Scenario parameters must be a dict.")

    return ScenarioSpec(
        scenario_id=scenario_id,
        name=name,
        description=description,
        type=stype,
        parameters=params,
    )


# ==================================================
# EXECUTIVE INTENT NORMALIZATION (PURE)
# ==================================================

def _intent_modifiers(intent: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convert executive intent into numeric modifiers.

    This is where choosing a recession ALWAYS introduces caution.
    """

    if not intent:
        return {"severity": 1.0, "duration": 1.0, "posture": 1.0}

    severity_map = {
        "light": 0.92,
        "moderate": 1.00,
        "severe": 1.25,
        "none": 1.0,
    }

    duration_map = {
        "short": 0.95,
        "medium": 1.00,
        "prolonged": 1.15,
    }

    posture_map = {
        "aggressive": 0.90,
        "balanced": 1.00,
        "defensive": 1.10,
    }

    return {
        "severity": severity_map.get(intent.get("severity", "none"), 1.0),
        "duration": duration_map.get(intent.get("duration", "medium"), 1.0),
        "posture": posture_map.get(intent.get("risk_posture", "balanced"), 1.0),
    }


# ==================================================
# CORE API
# ==================================================

def apply_scenario(
    baseline_df: pd.DataFrame,
    scenario: Dict[str, Any],
    *,
    executive_intent: Optional[Dict[str, Any]] = None,
    apply_to_intervals: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a deterministic scenario transform with executive intent modulation.
    """

    if not isinstance(baseline_df, pd.DataFrame):
        raise ValueError("baseline_df must be a DataFrame.")

    required = {"date", "forecast", "is_future"}
    if not required.issubset(baseline_df.columns):
        raise ValueError("baseline_df missing required columns.")

    spec = validate_scenario(scenario)
    intent_mod = _intent_modifiers(executive_intent)

    df = baseline_df.copy(deep=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Metadata (additive only)
    df["scenario_id"] = spec.scenario_id
    df["scenario_name"] = spec.name
    df["scenario_type"] = spec.type

    future_mask = df["is_future"].astype(bool)
    horizon = int(future_mask.sum())

    if spec.type == "identity" or horizon == 0:
        return df, _build_audit(spec, intent_mod, "identity")

    multipliers, details = _scenario_multipliers(
        spec=spec,
        horizon=horizon,
        intent_mod=intent_mod,
    )

    df.loc[future_mask, "forecast"] = _safe_multiply(
        df.loc[future_mask, "forecast"], multipliers
    )

    if apply_to_intervals and {"lower", "upper"}.issubset(df.columns):
        df.loc[future_mask, "lower"] = _safe_multiply(
            df.loc[future_mask, "lower"], multipliers
        )
        df.loc[future_mask, "upper"] = _safe_multiply(
            df.loc[future_mask, "upper"], multipliers
        )

    return df, _build_audit(spec, intent_mod, "intent-adjusted")


# ==================================================
# TRANSFORMS
# ==================================================

def _scenario_multipliers(
    *,
    spec: ScenarioSpec,
    horizon: int,
    intent_mod: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Any]]:

    p = spec.parameters
    sev = intent_mod["severity"]
    dur = intent_mod["duration"]
    pos = intent_mod["posture"]

    if spec.type == "level_shift":
        base = 1.0 + p["pct"] / 100.0
        return (
            np.full(horizon, base * sev * pos),
            {"base_pct": p["pct"], "intent_mod": intent_mod},
        )

    if spec.type == "trend_change":
        start = 1.0 + p["start_pct"] / 100.0
        end = 1.0 + p["end_pct"] / 100.0
        return (
            np.linspace(start, end * sev * pos, horizon),
            {"intent_mod": intent_mod},
        )

    if spec.type == "shock":
        base = 1.0 + p["magnitude_pct"] / 100.0
        recovery = max(1, int(p["recovery_periods"] * dur))
        decay = np.linspace(base * sev, 1.0, recovery)
        mult = np.ones(horizon)
        mult[: min(horizon, len(decay))] = decay[: min(horizon, len(decay))]
        return mult * pos, {
            "magnitude_pct": p["magnitude_pct"],
            "recovery_periods": recovery,
            "intent_mod": intent_mod,
        }

    return np.ones(horizon), {"intent_mod": intent_mod}


def _safe_multiply(series: pd.Series, multipliers: np.ndarray) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if len(s) != len(multipliers):
        raise ValueError("Multiplier length mismatch.")
    return pd.Series(s.values * multipliers, index=series.index)


# ==================================================
# AUDIT
# ==================================================

def _build_audit(
    spec: ScenarioSpec,
    intent_mod: Dict[str, float],
    transform: str,
) -> Dict[str, Any]:
    return {
        "scenario_id": spec.scenario_id,
        "scenario_name": spec.name,
        "scenario_type": spec.type,
        "scenario_description": spec.description,
        "transform": transform,
        "intent_modifiers": intent_mod,
    }
