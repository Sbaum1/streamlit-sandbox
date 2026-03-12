# ==================================================
# FILE: sentinel_engine/models/ets.py
# VERSION: 2.0.0
# MODEL: ETS (Additive Trend, No Seasonality)
# ENGINE: Sentinel Engine v2.0.0
# STATUS: VEDUTA ENGINE — REAL CI — 3B-2
# ==================================================
#
# 3B-2 CI FIX:
#   Previous: ci_low = np.nan, ci_high = np.nan (hard-coded)
#   Fixed:    Residual-based prediction intervals
#
#   Method: In-sample residuals → pooled std → horizon-scaled CI
#     residuals = y - fitted_values
#     sigma     = std(residuals)
#     sigma_h   = sigma * sqrt(h)   (error accumulation over horizon)
#     ci_low    = forecast - z * sigma_h
#     ci_high   = forecast + z * sigma_h
#
#   This is the standard approach for deterministic exponential
#   smoothing models and matches statsmodels internal simulation.
#   z-score mapped from confidence_level (0.90 → 1.645, etc.)
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sentinel_engine.models.contracts import ForecastResult


# --------------------------------------------------
# Z-SCORE MAP
# --------------------------------------------------

Z_SCORES = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def _z_score(confidence_level: float) -> float:
    return min(Z_SCORES.items(), key=lambda kv: abs(kv[0] - confidence_level))[1]


# --------------------------------------------------
# RESIDUAL-BASED CI
# --------------------------------------------------

def _residual_ci(
    forecast_values: np.ndarray,
    residuals:       np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals from in-sample residuals.

    sigma_h = sigma * sqrt(h) — standard error accumulation
    over forecast horizon for additive error models.
    """
    z      = _z_score(confidence_level)
    sigma  = float(np.std(residuals, ddof=1))
    h      = np.arange(1, len(forecast_values) + 1, dtype="float64")
    spread = z * sigma * np.sqrt(h)

    ci_low  = forecast_values - spread
    ci_high = forecast_values + spread

    return ci_low, ci_high


# ==================================================
# MODEL RUNNER
# ==================================================

def run_ets(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("ETS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"ETS requires monthly frequency. Detected: {inferred}")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(y) < 6:
        raise ValueError("Minimum 6 observations required.")

    # --------------------------------------------------
    # MODEL FIT
    # --------------------------------------------------

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit(
        optimized=False,
        smoothing_level=0.3,
        smoothing_trend=0.1,
    )

    hist_fitted = model.fittedvalues.astype("float64")

    if hist_fitted.isna().any():
        raise RuntimeError("NaN in fitted values.")

    # ── Residual-based CI (3B-2) ─────────────────────────────────────────────
    residuals = (y.values - hist_fitted.values).astype("float64")

    hist_block = pd.DataFrame(
        {
            "date":      hist_fitted.index,
            "actual":    np.nan,
            "forecast":  hist_fitted.values,
            "ci_low":    np.nan,
            "ci_mid":    hist_fitted.values,
            "ci_high":   np.nan,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_index = pd.date_range(
        start=hist_fitted.index[-1],
        periods=horizon + 1,
        freq=inferred,
    )[1:]

    if not future_index.min() > hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    future_forecast = model.forecast(horizon).astype("float64")

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite forecast values detected.")

    # ── Apply residual CI to future block ────────────────────────────────────
    ci_low, ci_high = _residual_ci(
        forecast_values  = future_forecast.values,
        residuals        = residuals,
        confidence_level = confidence_level,
    )

    future_block = pd.DataFrame(
        {
            "date":      future_index,
            "actual":    np.nan,
            "forecast":  future_forecast.values,
            "ci_low":    ci_low,
            "ci_mid":    future_forecast.values,
            "ci_high":   ci_high,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # DTYPE GOVERNANCE
    # --------------------------------------------------

    numeric_cols = ["forecast", "ci_low", "ci_mid", "ci_high"]
    hist_block[numeric_cols]   = hist_block[numeric_cols].astype("float64")
    future_block[numeric_cols] = future_block[numeric_cols].astype("float64")

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "ETS",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "trend":            "additive",
            "seasonality":      "none",
            "smoothing_level":  0.3,
            "smoothing_trend":  0.1,
            "frequency":        inferred,
            "confidence_level": confidence_level,
            "ci_method":        "residual_based_sigma_sqrt_h",
            "output_contract":       "ForecastResult",
        },
    )
