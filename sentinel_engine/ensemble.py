# ==================================================
# FILE: sentinel_engine/ensemble.py
# VERSION: 3.2.0
# ROLE: PRIMARY ENSEMBLE — MASE-WEIGHTED AGGREGATION
# ENGINE: Sentinel Engine v2.0.0
# UPDATED: G3 — Median-fallback weight bridge support
# UPDATED: G9 — Tier-filtered ensemble dispatch
# ==================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .contracts import ForecastResult, ENGINE_VERSION

MINIMUM_ENSEMBLE_QUORUM  = 2
MASE_FLOOR               = 1e-6
MASE_CAP                 = 10.0
ARIMA_FAMILY_CAP         = 0.40
MASE_EXCLUSION_THRESHOLD = 2.0
INTERMITTENT_ZERO_PCT    = 0.30

MODEL_FAMILY: Dict[str, str] = {
    "SARIMA":    "arima",
    "STL+ETS":   "arima",
    "HW_Damped": "arima",
    "Theta":     "arima",
    "DHR":       "arima",
    "ETS":       "ets",
    "TBATS":     "ets",
    "BSTS":      "bayesian",
    "LightGBM":  "ml",
    "NNETAR":    "ml",
    "GARCH":     "volatility",
    "Prophet":   "decomposition",
}

def _is_intermittent(df: pd.DataFrame) -> bool:
    if "value" not in df.columns:
        return False
    y = df["value"].astype("float64").values
    if len(y) == 0:
        return False
    return float((y == 0).mean()) > INTERMITTENT_ZERO_PCT

def _compute_mase_weights(
    member_names: List[str],
    member_metrics: Dict[str, dict],
) -> Tuple[Dict[str, float], str]:
    raw_weights: Dict[str, float] = {}
    for name in member_names:
        metrics = member_metrics.get(name, {})
        mase    = metrics.get("MASE") or metrics.get("mase")
        if mase is None or not np.isfinite(mase) or mase <= 0:
            equal = 1.0 / len(member_names)
            return {n: equal for n in member_names}, "simple_mean_fallback"
        raw_weights[name] = 1.0 / float(np.clip(mase, MASE_FLOOR, MASE_CAP))
    total = sum(raw_weights.values())
    if total <= 0:
        equal = 1.0 / len(member_names)
        return {n: equal for n in member_names}, "simple_mean_fallback"
    return {n: w / total for n, w in raw_weights.items()}, "mase_weighted"

def _apply_mase_exclusion(
    member_names: List[str],
    pre_computed_weights: Optional[Dict[str, float]],
    member_metrics: Dict[str, dict],
    threshold: float,
) -> Tuple[List[str], List[str]]:
    excluded = []
    active   = []
    for name in member_names:
        metrics = member_metrics.get(name, {})
        mase    = metrics.get("MASE") or metrics.get("mase")
        if mase is not None and np.isfinite(mase) and mase > threshold:
            excluded.append(name)
        else:
            active.append(name)
    return active, excluded

def _apply_family_diversity_cap(
    weights: Dict[str, float],
    family_map: Dict[str, str],
    arima_cap: float,
) -> Tuple[Dict[str, float], bool]:
    arima_members     = [n for n in weights if family_map.get(n) == "arima"]
    non_arima_members = [n for n in weights if family_map.get(n) != "arima"]
    arima_total       = sum(weights[n] for n in arima_members)

    if arima_total <= arima_cap or not non_arima_members:
        return weights, False

    scale_factor = arima_cap / arima_total
    adjusted = dict(weights)
    for n in arima_members:
        adjusted[n] = weights[n] * scale_factor

    excess          = arima_total - arima_cap
    non_arima_total = sum(weights[n] for n in non_arima_members)

    if non_arima_total > 0:
        for n in non_arima_members:
            adjusted[n] = weights[n] + excess * (weights[n] / non_arima_total)
    else:
        share = excess / len(non_arima_members)
        for n in non_arima_members:
            adjusted[n] = weights.get(n, 0.0) + share

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {n: w / total for n, w in adjusted.items()}

    return adjusted, True

def _execute_members(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
    active_tier: str = "enterprise",
) -> Tuple[Dict[str, ForecastResult], List[str]]:
    from .registry import get_ensemble_members_by_tier
    members  = get_ensemble_members_by_tier(active_tier)
    component_results: Dict[str, ForecastResult] = {}
    excluded: List[str] = []

    for entry in sorted(members, key=lambda e: e["name"]):
        name   = entry["name"]
        runner = entry["runner"]

        if name == "VAR":
            numeric_cols = [c for c in df.columns
                            if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 2:
                excluded.append(name)
                continue

        if name in ("Croston_SBA", "Croston_Classic"):
            if not _is_intermittent(df):
                excluded.append(name)
                continue

        try:
            result = runner(df=df, horizon=horizon, confidence_level=confidence_level)
        except Exception:
            excluded.append(name)
            continue

        if not isinstance(result, ForecastResult):
            excluded.append(name)
            continue
        if result.forecast_df is None or result.forecast_df.empty:
            excluded.append(name)
            continue

        try:
            forecast_values = result.forecast_df["forecast"].astype(float).values
        except (TypeError, ValueError):
            excluded.append(name)
            continue

        if not np.isfinite(forecast_values).all():
            excluded.append(name)
            continue

        component_results[name] = result

    return component_results, excluded

def _extract_future_blocks(
    component_results: Dict[str, ForecastResult],
    last_observed: pd.Timestamp,
    horizon: int,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    future_blocks: Dict[str, pd.DataFrame] = {}
    excluded: List[str] = []

    for name in sorted(component_results.keys()):
        result      = component_results[name]
        forecast_df = result.forecast_df.copy()
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        for col in ["forecast", "ci_low", "ci_high"]:
            if col in forecast_df.columns:
                forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

        future_block = forecast_df.loc[
            forecast_df["date"] > last_observed
        ].copy().reset_index(drop=True)

        if len(future_block) != horizon:
            excluded.append(name)
            continue
        if not np.isfinite(future_block["forecast"].values.astype(float)).all():
            excluded.append(name)
            continue

        future_blocks[name] = future_block[["date", "forecast", "ci_low", "ci_high"]]

    return future_blocks, excluded

def _compute_ci_from_spread(
    forecast_matrix: np.ndarray,
    confidence_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    z_scores = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z    = min(z_scores.items(), key=lambda kv: abs(kv[0] - confidence_level))[1]
    std  = np.std(forecast_matrix,  axis=0)
    mean = np.mean(forecast_matrix, axis=0)
    return mean - z * std, mean + z * std

def _aggregate(
    future_blocks: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    reference_dates: np.ndarray,
    confidence_level: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    weighted_forecasts: List[np.ndarray] = []
    weighted_ci_low:    List[np.ndarray] = []
    weighted_ci_high:   List[np.ndarray] = []
    raw_forecasts:      List[np.ndarray] = []

    for name, block in future_blocks.items():
        if not np.array_equal(block["date"].values, reference_dates):
            continue
        w             = weights.get(name, 0.0)
        forecast_vals = block["forecast"].values.astype(float)
        ci_low_vals   = block["ci_low"].values.astype(float)
        ci_high_vals  = block["ci_high"].values.astype(float)
        weighted_forecasts.append(forecast_vals * w)
        weighted_ci_low.append(ci_low_vals      * w)
        weighted_ci_high.append(ci_high_vals    * w)
        raw_forecasts.append(forecast_vals)

    if not weighted_forecasts:
        raise RuntimeError("Ensemble aggregation failed — no aligned members.")

    ensemble_forecast    = np.sum(np.vstack(weighted_forecasts), axis=0)
    weighted_ci_low_sum  = np.sum(np.vstack(weighted_ci_low),    axis=0)
    weighted_ci_high_sum = np.sum(np.vstack(weighted_ci_high),   axis=0)
    ci_method = "weighted_model_ci"

    if (
        not np.isfinite(weighted_ci_low_sum).all()
        or not np.isfinite(weighted_ci_high_sum).all()
        or (weighted_ci_high_sum <= weighted_ci_low_sum).any()
    ):
        fm = np.vstack(raw_forecasts)
        weighted_ci_low_sum, weighted_ci_high_sum = _compute_ci_from_spread(
            fm, confidence_level
        )
        ci_method = "spread_based_ci"

    return ensemble_forecast, weighted_ci_low_sum, weighted_ci_high_sum, ci_method


def run_primary_ensemble(
    df:                   pd.DataFrame,
    horizon:              int,
    confidence_level:     float,
    pre_computed_weights: Optional[Dict[str, float]] = None,
    active_tier:          str = "enterprise",
) -> ForecastResult:
    if df is None or df.empty:
        raise ValueError("Primary Ensemble received empty dataframe.")

    from .registry import get_ensemble_members_by_tier
    last_observed          = pd.to_datetime(df["date"]).max()
    total_registry_members = len(get_ensemble_members_by_tier(active_tier))

    # PHASE 3D: Croston routing
    intermittent = _is_intermittent(df)
    if intermittent:
        from .models.croston import run_croston
        try:
            croston_result = run_croston(
                df=df, horizon=horizon,
                confidence_level=confidence_level, variant="sba",
            )
            fc_df = croston_result.forecast_df.copy()
            meta  = croston_result.metadata.copy()
            meta["routing"]           = "croston_intermittent"
            meta["zero_pct_detected"] = float(
                (df["value"].astype("float64").values == 0).mean()
            )
            meta["ensemble_note"] = (
                "Series routed to Croston_SBA due to intermittent demand. "
                "Standard ensemble bypassed."
            )
            return ForecastResult(
                model_name  = "Primary Ensemble",
                forecast_df = fc_df[[
                    "date", "actual", "forecast",
                    "ci_low", "ci_mid", "ci_high", "error_pct"
                ]],
                metrics  = {},
                metadata = meta,
            )
        except Exception:
            intermittent = False

    # Execute members — tier-filtered (G9)
    component_results, excluded_execution = _execute_members(
        df=df, horizon=horizon, confidence_level=confidence_level,
        active_tier=active_tier,
    )
    future_blocks, excluded_extraction = _extract_future_blocks(
        component_results=component_results,
        last_observed=last_observed,
        horizon=horizon,
    )

    # PHASE 3D: Auto exclusion
    member_names   = sorted(future_blocks.keys())
    member_metrics = {n: (component_results[n].metrics or {}) for n in member_names}
    active_names, excluded_mase = _apply_mase_exclusion(
        member_names=member_names,
        pre_computed_weights=pre_computed_weights,
        member_metrics=member_metrics,
        threshold=MASE_EXCLUSION_THRESHOLD,
    )
    for name in excluded_mase:
        future_blocks.pop(name, None)

    all_excluded = sorted(set(excluded_execution + excluded_extraction + excluded_mase))
    valid_count  = len(future_blocks)

    if valid_count < MINIMUM_ENSEMBLE_QUORUM:
        raise RuntimeError(
            f"Primary Ensemble quorum failure — "
            f"{valid_count} valid member(s), minimum {MINIMUM_ENSEMBLE_QUORUM} required. "
            f"Excluded: {all_excluded}"
        )

    active_names   = sorted(future_blocks.keys())
    member_metrics = {n: (component_results[n].metrics or {}) for n in active_names}

    # Weight computation with per-tier normalisation
    median_fallback_models: List[str] = []
    if pre_computed_weights is not None:
        # G3: Strip sentinel key inserted by runner's median-fallback bridge
        median_fallback_models = pre_computed_weights.pop(  # type: ignore[arg-type]
            "_median_fallback_models", []
        )
        raw = {
            n: pre_computed_weights[n]
            for n in active_names
            if n in pre_computed_weights and pre_computed_weights[n] > 0
        }
        if len(raw) >= MINIMUM_ENSEMBLE_QUORUM:
            total = sum(raw.values())
            weights = (
                {n: w / total for n, w in raw.items()}
                if total > 0
                else {n: 1.0 / len(raw) for n in raw}
            )
            aggregation_method = (
                "mase_weighted_partial" if median_fallback_models
                else "mase_weighted"
            )
        else:
            weights, aggregation_method = _compute_mase_weights(active_names, member_metrics)
    else:
        weights, aggregation_method = _compute_mase_weights(active_names, member_metrics)

    # PHASE 3D: Family diversity cap
    weights, cap_applied = _apply_family_diversity_cap(
        weights=weights, family_map=MODEL_FAMILY, arima_cap=ARIMA_FAMILY_CAP,
    )
    if cap_applied:
        aggregation_method = aggregation_method + "_diversity_capped"

    reference_dates = future_blocks[active_names[0]]["date"].values

    ensemble_forecast, ensemble_ci_low, ensemble_ci_high, ci_method = _aggregate(
        future_blocks=future_blocks,
        weights=weights,
        reference_dates=reference_dates,
        confidence_level=confidence_level,
    )

    if not np.isfinite(ensemble_forecast).all():
        raise RuntimeError("Primary Ensemble produced non-finite values.")

    family_weights: Dict[str, float] = {}
    for name, w in weights.items():
        fam = MODEL_FAMILY.get(name, "other")
        family_weights[fam] = family_weights.get(fam, 0.0) + w

    ensemble_df = pd.DataFrame({
        "date":      reference_dates,
        "actual":    pd.NA,
        "forecast":  ensemble_forecast,
        "ci_low":    ensemble_ci_low,
        "ci_mid":    ensemble_forecast,
        "ci_high":   ensemble_ci_high,
        "error_pct": pd.NA,
    })

    metadata = {
        "engine_version":           ENGINE_VERSION,
        "aggregation_method":       aggregation_method,
        "ci_method":                ci_method,
        "component_count_total":    total_registry_members,
        "component_count_valid":    valid_count,
        "excluded_components":      all_excluded,
        "excluded_mase_threshold":  excluded_mase,
        "member_weights":           {n: round(w, 6) for n, w in weights.items()},
        "family_weights":           {f: round(w, 6) for f, w in family_weights.items()},
        "arima_family_cap":         ARIMA_FAMILY_CAP,
        "diversity_cap_applied":    cap_applied,
        "mase_exclusion_threshold": MASE_EXCLUSION_THRESHOLD,
        "intermittent_routing":     intermittent,
        "shock_guard_enabled":      True,
        "quorum_minimum":           MINIMUM_ENSEMBLE_QUORUM,
        "confidence_level":         confidence_level,
        "weights_source":           (
            "runner_backtest_mase" if pre_computed_weights else "internal"
        ),
        "median_fallback_models":   median_fallback_models,
        "active_tier":              active_tier,
    }

    return ForecastResult(
        model_name  = "Primary Ensemble",
        forecast_df = ensemble_df[[
            "date", "actual", "forecast",
            "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = {},
        metadata = metadata,
    )
