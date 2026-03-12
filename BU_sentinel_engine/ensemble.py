# ==================================================
# FILE: sentinel_engine/ensemble.py
# VERSION: 2.0.0
# ROLE: PRIMARY ENSEMBLE — MASE-WEIGHTED AGGREGATION
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - Deterministic execution order (sorted model names)
# - MASE-weighted blending — primary aggregation method
# - Simple mean fallback — activates when MASE unavailable
# - Shock guard — non-finite values excluded before aggregation
# - Empty output guard — models with no future rows excluded
# - Horizon guard — models with wrong horizon length excluded
# - Minimum quorum — raises RuntimeError if < 2 valid members
# - Never mutates input DataFrames
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .contracts import ForecastResult, ENGINE_VERSION
from .registry  import get_ensemble_members


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MINIMUM_ENSEMBLE_QUORUM = 2        # Minimum valid members to produce output
MASE_FLOOR              = 1e-6     # Prevents division by zero in weight calc
MASE_CAP                = 10.0     # Caps runaway MASE before weight inversion


# ==================================================
# MASE WEIGHT COMPUTATION
# ==================================================

def _compute_mase_weights(
    member_names:   List[str],
    member_metrics: Dict[str, dict],
) -> Dict[str, float]:
    """
    Compute normalized inverse-MASE weights for ensemble blending.

    Logic:
    - Lower MASE = better model = higher weight
    - Weight = 1 / MASE (capped and floored for stability)
    - Weights are normalized to sum to 1.0
    - Falls back to equal weights if MASE unavailable for any member

    Returns:
        Dict mapping model name -> normalized weight (sums to 1.0)
    """

    raw_weights: Dict[str, float] = {}

    for name in member_names:

        metrics = member_metrics.get(name, {})
        mase    = metrics.get("MASE") or metrics.get("mase")

        if mase is None or not np.isfinite(mase) or mase <= 0:
            # MASE unavailable — fall back to equal weighting for all members
            equal_weight = 1.0 / len(member_names)
            return {n: equal_weight for n in member_names}

        mase_clamped       = float(np.clip(mase, MASE_FLOOR, MASE_CAP))
        raw_weights[name]  = 1.0 / mase_clamped

    total = sum(raw_weights.values())

    if total <= 0:
        equal_weight = 1.0 / len(member_names)
        return {n: equal_weight for n in member_names}

    return {name: w / total for name, w in raw_weights.items()}


# ==================================================
# COMPONENT EXECUTION
# ==================================================

def _execute_members(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> Tuple[Dict[str, ForecastResult], List[str]]:
    """
    Execute all ensemble member models in deterministic order.

    Returns:
        component_results : Dict of name -> ForecastResult for valid members
        excluded          : List of names excluded due to failure or bad output
    """

    members            = get_ensemble_members()
    component_results: Dict[str, ForecastResult] = {}
    excluded:          List[str]                 = []

    # Deterministic execution order
    for entry in sorted(members, key=lambda e: e["name"]):

        name   = entry["name"]
        runner = entry["runner"]

        try:
            result = runner(
                df               = df,
                horizon          = horizon,
                confidence_level = confidence_level,
            )
        except Exception:
            excluded.append(name)
            continue

        # Contract check
        if not isinstance(result, ForecastResult):
            excluded.append(name)
            continue

        # Empty output guard
        if result.forecast_df is None or result.forecast_df.empty:
            excluded.append(name)
            continue

        # Non-finite guard
        forecast_values = result.forecast_df["forecast"].values
        if not np.isfinite(forecast_values).all():
            excluded.append(name)
            continue

        component_results[name] = result

    return component_results, excluded


# ==================================================
# FUTURE BLOCK EXTRACTION
# ==================================================

def _extract_future_blocks(
    component_results: Dict[str, ForecastResult],
    last_observed:     pd.Timestamp,
    horizon:           int,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Extract and validate future forecast blocks from each member result.

    Returns:
        future_blocks : Dict of name -> future-only DataFrame
        excluded      : Names excluded due to horizon mismatch or bad values
    """

    future_blocks: Dict[str, pd.DataFrame] = {}
    excluded:      List[str]               = []

    for name in sorted(component_results.keys()):

        result      = component_results[name]
        forecast_df = result.forecast_df.copy()
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        future_block = forecast_df.loc[
            forecast_df["date"] > last_observed
        ].copy().reset_index(drop=True)

        # Horizon guard
        if len(future_block) != horizon:
            excluded.append(name)
            continue

        # Non-finite guard on future block specifically
        if not np.isfinite(future_block["forecast"].values).all():
            excluded.append(name)
            continue

        future_blocks[name] = future_block[
            ["date", "forecast", "ci_low", "ci_high"]
        ]

    return future_blocks, excluded


# ==================================================
# WEIGHTED AGGREGATION
# ==================================================

def _aggregate(
    future_blocks: Dict[str, pd.DataFrame],
    weights:       Dict[str, float],
    reference_dates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply MASE weights and aggregate forecast + CI arrays.

    Returns:
        ensemble_forecast : weighted mean forecast array
        ensemble_ci_low   : weighted mean lower CI array
        ensemble_ci_high  : weighted mean upper CI array
    """

    weighted_forecasts: List[np.ndarray] = []
    weighted_ci_low:    List[np.ndarray] = []
    weighted_ci_high:   List[np.ndarray] = []

    for name, block in future_blocks.items():

        # Date alignment check
        if not np.array_equal(block["date"].values, reference_dates):
            continue

        w = weights.get(name, 0.0)

        weighted_forecasts.append(block["forecast"].values * w)
        weighted_ci_low.append(block["ci_low"].values    * w)
        weighted_ci_high.append(block["ci_high"].values  * w)

    if not weighted_forecasts:
        raise RuntimeError(
            "Ensemble aggregation failed — no aligned members after weighting."
        )

    ensemble_forecast = np.sum(np.vstack(weighted_forecasts), axis=0)
    ensemble_ci_low   = np.sum(np.vstack(weighted_ci_low),    axis=0)
    ensemble_ci_high  = np.sum(np.vstack(weighted_ci_high),   axis=0)

    return ensemble_forecast, ensemble_ci_low, ensemble_ci_high


# ==================================================
# PRIMARY ENSEMBLE RUNNER
# ==================================================

def run_primary_ensemble(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:
    """
    Sentinel Engine v2.0.0 Primary Ensemble.

    Aggregation:    MASE-weighted blending (inverse-MASE normalized weights)
    Fallback:       Simple mean (equal weights) if MASE unavailable
    Shock guard:    Non-finite outputs excluded before aggregation
    Quorum:         Minimum 2 valid members required
    Order:          Deterministic — sorted member names

    Args:
        df               : Historical DataFrame with 'date' and 'value' columns
        horizon          : Number of periods to forecast
        confidence_level : Confidence level for prediction intervals (e.g. 0.9)

    Returns:
        ForecastResult with ensemble forecast, weighted CI, and full metadata

    Raises:
        ValueError  : Empty or None input DataFrame
        RuntimeError: Fewer than MINIMUM_ENSEMBLE_QUORUM valid members
    """

    if df is None or df.empty:
        raise ValueError("Primary Ensemble received empty dataframe.")

    last_observed = pd.to_datetime(df["date"]).max()

    # ── Step 1: Execute all ensemble members ────────────────────────────────
    component_results, excluded_execution = _execute_members(
        df               = df,
        horizon          = horizon,
        confidence_level = confidence_level,
    )

    # ── Step 2: Extract future blocks ────────────────────────────────────────
    future_blocks, excluded_extraction = _extract_future_blocks(
        component_results = component_results,
        last_observed     = last_observed,
        horizon           = horizon,
    )

    all_excluded = sorted(set(excluded_execution + excluded_extraction))

    # ── Step 3: Quorum check ─────────────────────────────────────────────────
    valid_count = len(future_blocks)

    if valid_count < MINIMUM_ENSEMBLE_QUORUM:
        raise RuntimeError(
            f"Primary Ensemble quorum failure — "
            f"{valid_count} valid member(s), minimum {MINIMUM_ENSEMBLE_QUORUM} required. "
            f"Excluded: {all_excluded}"
        )

    # ── Step 4: Collect member metrics for MASE weighting ───────────────────
    member_metrics: Dict[str, dict] = {
        name: (component_results[name].metrics or {})
        for name in future_blocks.keys()
    }

    # ── Step 5: Compute MASE weights ─────────────────────────────────────────
    member_names   = sorted(future_blocks.keys())
    weights        = _compute_mase_weights(member_names, member_metrics)
    aggregation_method = (
        "mase_weighted"
        if any(
            (component_results[n].metrics or {}).get("MASE") is not None
            for n in member_names
        )
        else "simple_mean_fallback"
    )

    # ── Step 6: Aggregate ────────────────────────────────────────────────────
    reference_dates                          = future_blocks[member_names[0]]["date"].values
    ensemble_forecast, ensemble_ci_low, ensemble_ci_high = _aggregate(
        future_blocks   = future_blocks,
        weights         = weights,
        reference_dates = reference_dates,
    )

    # ── Step 7: Final non-finite guard ───────────────────────────────────────
    if not np.isfinite(ensemble_forecast).all():
        raise RuntimeError(
            "Primary Ensemble produced non-finite values after aggregation."
        )

    # ── Step 8: Construct output DataFrame ───────────────────────────────────
    ensemble_df = pd.DataFrame({
        "date":      reference_dates,
        "actual":    pd.NA,
        "forecast":  ensemble_forecast,
        "ci_low":    ensemble_ci_low,
        "ci_mid":    ensemble_forecast,
        "ci_high":   ensemble_ci_high,
        "error_pct": pd.NA,
    })

    # ── Step 9: Build metadata ───────────────────────────────────────────────
    metadata = {
        "engine_version":           ENGINE_VERSION,
        "aggregation_method":       aggregation_method,
        "component_count_total":    len(get_ensemble_members()),
        "component_count_valid":    valid_count,
        "excluded_components":      all_excluded,
        "member_weights":           {
            name: round(w, 6) for name, w in weights.items()
        },
        "shock_guard_enabled":      True,
        "quorum_minimum":           MINIMUM_ENSEMBLE_QUORUM,
        "confidence_level":         confidence_level,
    }

    return ForecastResult(
        model_name   = "Primary Ensemble",
        forecast_df  = ensemble_df[[
            "date", "actual", "forecast",
            "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics      = {},
        metadata     = metadata,
    )