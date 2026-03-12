# ==================================================
# FILE: certification/bootstrap_ci.py
# ROLE: CENTRALIZED RESIDUAL BOOTSTRAP CI WRAPPER
# PURPOSE: Upgrade deterministic models without CI
# STATUS: EXECUTIVE-GRADE / DETERMINISTIC / AUDIT-SAFE
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from models.contracts import ForecastResult


def apply_bootstrap_ci(
    model_runner: Callable,
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> ForecastResult:
    """
    Apply residual bootstrap prediction intervals to models
    that do not natively supply confidence intervals.

    Rules:
    - Does NOT modify model logic
    - Only applies if CI columns are NA
    - Deterministic via fixed seed
    - Future rows only are modified
    """

    # --------------------------------------------------
    # Run Base Model
    # --------------------------------------------------

    base_result = model_runner(df, horizon, confidence_level)

    forecast_df = base_result.forecast_df.copy()

    if forecast_df.empty:
        raise RuntimeError("Bootstrap CI cannot be applied to empty forecast output.")

    # --------------------------------------------------
    # Detect Existing CI (Refuse Override)
    # --------------------------------------------------

    future_mask = forecast_df["forecast"].notna() & forecast_df["ci_low"].isna()

    if forecast_df["ci_low"].notna().any():
        # Model already provides CI — do not override
        return base_result

    # --------------------------------------------------
    # Separate Historical vs Future Rows
    # --------------------------------------------------

    historical_mask = forecast_df["ci_mid"].notna() & forecast_df["ci_low"].isna()

    historical_block = forecast_df.loc[historical_mask].copy()
    future_block = forecast_df.loc[~historical_mask].copy()

    if len(future_block) != horizon:
        raise RuntimeError("Future horizon length mismatch in bootstrap CI wrapper.")

    # --------------------------------------------------
    # Reconstruct Actual Series for Residual Calculation
    # --------------------------------------------------

    df_sorted = df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date")

    actual_series = df_sorted.set_index("date")["value"].astype(float)

    fitted_series = pd.Series(
        historical_block["ci_mid"].values,
        index=pd.to_datetime(historical_block["date"]),
        dtype="float64",
    )

    if not fitted_series.index.equals(actual_series.index):
        raise RuntimeError("Historical index misalignment during bootstrap CI.")

    residuals = actual_series - fitted_series

    if residuals.isna().any():
        raise RuntimeError("NaN residuals detected.")

    if not np.isfinite(residuals).all():
        raise RuntimeError("Non-finite residuals detected.")

    if len(residuals) < 12:
        raise RuntimeError("Insufficient residual history for bootstrap CI.")

    if np.var(residuals) == 0:
        raise RuntimeError("Residual variance is zero; bootstrap CI invalid.")

    # --------------------------------------------------
    # Bootstrap Simulation
    # --------------------------------------------------

    rng = np.random.default_rng(seed)

    point_forecast = future_block["forecast"].values
    simulations = np.zeros((n_simulations, horizon), dtype="float64")

    residual_array = residuals.values

    for i in range(n_simulations):
        sampled_residuals = rng.choice(residual_array, size=horizon, replace=True)
        simulations[i, :] = point_forecast + sampled_residuals

    # --------------------------------------------------
    # Empirical Interval Construction
    # --------------------------------------------------

    alpha = 1.0 - confidence_level

    ci_low = np.percentile(simulations, 100 * (alpha / 2.0), axis=0)
    ci_high = np.percentile(simulations, 100 * (1.0 - alpha / 2.0), axis=0)

    if np.any(ci_low > ci_high):
        raise RuntimeError("Bootstrap CI inversion detected.")

    # --------------------------------------------------
    # Inject CI into Future Block
    # --------------------------------------------------

    future_block.loc[:, "ci_low"] = ci_low
    future_block.loc[:, "ci_high"] = ci_high
    future_block.loc[:, "ci_mid"] = future_block["forecast"].values

    # --------------------------------------------------
    # Determinism Gate (Re-run Once)
    # --------------------------------------------------

    rng_test = np.random.default_rng(seed)
    simulations_test = np.zeros((n_simulations, horizon), dtype="float64")

    for i in range(n_simulations):
        sampled_residuals = rng_test.choice(residual_array, size=horizon, replace=True)
        simulations_test[i, :] = point_forecast + sampled_residuals

    ci_low_test = np.percentile(simulations_test, 100 * (alpha / 2.0), axis=0)
    ci_high_test = np.percentile(simulations_test, 100 * (1.0 - alpha / 2.0), axis=0)

    if (
        np.max(np.abs(ci_low - ci_low_test)) > 1e-9
        or np.max(np.abs(ci_high - ci_high_test)) > 1e-9
    ):
        raise RuntimeError("Bootstrap CI determinism failure detected.")

    # --------------------------------------------------
    # Reassemble Forecast Output
    # --------------------------------------------------

    upgraded_df = pd.concat([historical_block, future_block], ignore_index=True)
    upgraded_df = upgraded_df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # Metadata Update
    # --------------------------------------------------

    updated_metadata = dict(base_result.metadata)
    updated_metadata.update(
        {
            "ci_method": "residual_bootstrap",
            "bootstrap_simulations": n_simulations,
            "ci_upgraded": True,
        }
    )

    return ForecastResult(
        model_name=base_result.model_name,
        forecast_df=upgraded_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=base_result.metrics,
        metadata=updated_metadata,
    )
