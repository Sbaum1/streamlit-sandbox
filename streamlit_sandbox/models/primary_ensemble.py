# ==================================================
# FILE: streamlit_sandbox/models/primary_ensemble.py
# ROLE: PRIMARY ENSEMBLE (HARDENED SHOCK GUARD)
# STANDARD: FORTUNE 100 / DETERMINISTIC
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from streamlit_sandbox.models.contracts import ForecastResult
from streamlit_sandbox.models.sarima import run_sarima
from streamlit_sandbox.models.stl_ets import run_stl_ets
from streamlit_sandbox.models.prophet_model import run_prophet


PRIMARY_MEMBERS = {
    "Prophet": run_prophet,
    "SARIMA": run_sarima,
    "STL+ETS": run_stl_ets,
}


def run_primary_ensemble(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    if df is None or df.empty:
        raise ValueError("Primary Ensemble received empty dataframe.")

    component_results: Dict[str, ForecastResult] = {}
    excluded_components = []

    # --------------------------------------------------
    # Execute Component Models (Deterministic Order)
    # --------------------------------------------------

    for name in sorted(PRIMARY_MEMBERS.keys()):
        runner = PRIMARY_MEMBERS[name]

        try:
            result = runner(
                df=df,
                horizon=horizon,
                confidence_level=confidence_level,
            )
        except Exception:
            excluded_components.append(name)
            continue

        if not isinstance(result, ForecastResult):
            excluded_components.append(name)
            continue

        if result.forecast_df is None or result.forecast_df.empty:
            excluded_components.append(name)
            continue

        if not np.isfinite(result.forecast_df["forecast"].values).all():
            excluded_components.append(name)
            continue

        component_results[name] = result

    component_count_total = len(PRIMARY_MEMBERS)
    component_count_valid = len(component_results)

    if component_count_valid == 0:
        raise RuntimeError("All ensemble components invalid")

    # --------------------------------------------------
    # Extract Future Forecast Blocks
    # --------------------------------------------------

    last_observed = pd.to_datetime(df["date"]).max()

    future_blocks = []

    for name in sorted(component_results.keys()):
        result = component_results[name]
        forecast_df = result.forecast_df.copy()
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        future_block = forecast_df.loc[
            forecast_df["date"] > last_observed
        ].copy()

        if len(future_block) != horizon:
            excluded_components.append(name)
            continue

        if not np.isfinite(future_block["forecast"].values).all():
            excluded_components.append(name)
            continue

        future_blocks.append(
            future_block[["date", "forecast", "ci_low", "ci_high"]].reset_index(drop=True)
        )

    if not future_blocks:
        raise RuntimeError("All ensemble components invalid")

    # --------------------------------------------------
    # Deterministic Alignment (Use First Valid Block)
    # --------------------------------------------------

    reference_dates = future_blocks[0]["date"].values

    aligned_forecasts = []
    aligned_ci_low = []
    aligned_ci_high = []

    for block in future_blocks:
        if not np.array_equal(block["date"].values, reference_dates):
            continue
        aligned_forecasts.append(block["forecast"].values)
        aligned_ci_low.append(block["ci_low"].values)
        aligned_ci_high.append(block["ci_high"].values)

    if not aligned_forecasts:
        raise RuntimeError("All ensemble components invalid")

    # --------------------------------------------------
    # Simple Mean Aggregation
    # --------------------------------------------------

    stacked_forecasts = np.vstack(aligned_forecasts)
    stacked_ci_low = np.vstack(aligned_ci_low)
    stacked_ci_high = np.vstack(aligned_ci_high)

    ensemble_forecast = np.mean(stacked_forecasts, axis=0)
    ensemble_ci_low = np.mean(stacked_ci_low, axis=0)
    ensemble_ci_high = np.mean(stacked_ci_high, axis=0)

    if not np.isfinite(ensemble_forecast).all():
        raise RuntimeError("Primary Ensemble produced non-finite values")

    # --------------------------------------------------
    # Construct Output
    # --------------------------------------------------

    ensemble_df = pd.DataFrame(
        {
            "date": reference_dates,
            "actual": pd.NA,
            "forecast": ensemble_forecast,
            "ci_low": ensemble_ci_low,
            "ci_mid": ensemble_forecast,
            "ci_high": ensemble_ci_high,
            "error_pct": pd.NA,
        }
    )

    metadata = {
        "component_count_total": component_count_total,
        "component_count_valid": len(aligned_forecasts),
        "excluded_components": sorted(set(excluded_components)),
        "aggregation_method": "simple_mean",
        "shock_guard_enabled": True,
    }

    return ForecastResult(
        model_name="Primary Ensemble",
        forecast_df=ensemble_df[
            [
                "date",
                "actual",
                "forecast",
                "ci_low",
                "ci_mid",
                "ci_high",
                "error_pct",
            ]
        ],
        metrics={},
        metadata=metadata,
    )