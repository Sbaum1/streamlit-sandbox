# ==================================================
# FILE: sentinel_engine/models/theta.py
# VERSION: 2.0.0
# MODEL: THETA (Additive / Deterministic)
# ENGINE: Sentinel Engine v2.0.0
# STATUS: VEDUTA ENGINE — REAL CI — 3B-2
# ==================================================
#
# 3B-2 CI FIX:
#   Previous: ci_low = pd.NA, ci_high = pd.NA (hard-coded)
#   Fixed:    Residual-based prediction intervals
#
#   Method: In-sample residuals → pooled std → horizon-scaled CI
#     residuals = y - theta_fitted
#     sigma     = std(residuals)
#     sigma_h   = sigma * sqrt(h)
#     ci_low    = forecast - z * sigma_h
#     ci_high   = forecast + z * sigma_h
#
#   Theta is a custom deterministic model (SES + linear trend).
#   No library CI is available. Residual-based intervals are
#   the standard approach for this class of model.
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from sentinel_engine.models.contracts import ForecastResult


EPSILON    = 1e-8
CLIP_LIMIT = 1e12


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


def _residual_ci(
    forecast_values:  np.ndarray,
    residuals:        np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    z      = _z_score(confidence_level)
    sigma  = float(np.std(residuals, ddof=1))
    h      = np.arange(1, len(forecast_values) + 1, dtype="float64")
    spread = z * sigma * np.sqrt(h)
    return forecast_values - spread, forecast_values + spread


# ==================================================
# MODEL RUNNER
# ==================================================

def run_theta(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Theta requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if inferred not in ("MS", "M"):
        raise ValueError("Theta requires monthly frequency.")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in input.")

    if len(y) < 12:
        raise ValueError("Minimum 12 observations required.")

    # --------------------------------------------------
    # THETA CORE
    # --------------------------------------------------

    t        = np.arange(len(y), dtype="float64")
    y_values = y.values.astype("float64")

    # Linear regression component (Theta=2 equivalent)
    slope     = np.sum((t - t.mean()) * (y_values - y_values.mean())) / (
                    np.sum((t - t.mean()) ** 2) + EPSILON
                )
    intercept = y_values.mean() - slope * t.mean()

    trend_component = intercept + slope * t

    # SES component (Theta=0 equivalent)
    ses_model  = SimpleExpSmoothing(y_values, initialization_method="estimated")
    ses_fit    = ses_model.fit(optimized=True)
    ses_fitted = ses_fit.fittedvalues.astype("float64")

    # Combine components
    theta_fitted = np.clip(
        0.5 * (trend_component + ses_fitted),
        -CLIP_LIMIT, CLIP_LIMIT
    )

    if not np.isfinite(theta_fitted).all():
        raise RuntimeError("Theta produced non-finite fitted values.")

    # ── Residuals for CI (3B-2) ──────────────────────────────────────────────
    residuals = (y_values - theta_fitted).astype("float64")

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_t = np.arange(len(y), len(y) + horizon, dtype="float64")

    trend_future = intercept + slope * future_t

    ses_last = ses_fitted[-1]
    if not np.isfinite(ses_last):
        ses_last = y_values[-1]

    ses_future      = np.full(horizon, ses_last, dtype="float64")
    forecast_values = np.clip(
        0.5 * (trend_future + ses_future),
        -CLIP_LIMIT, CLIP_LIMIT
    )

    hardened_fallback = False
    if not np.isfinite(forecast_values).all():
        fallback = np.clip(intercept + slope * future_t, -CLIP_LIMIT, CLIP_LIMIT)
        if not np.isfinite(fallback).all():
            raise RuntimeError("Theta produced non-finite values after hardening.")
        forecast_values   = fallback
        hardened_fallback = True

    # ── Residual CI (3B-2) ───────────────────────────────────────────────────
    ci_low, ci_high = _residual_ci(
        forecast_values  = forecast_values,
        residuals        = residuals,
        confidence_level = confidence_level,
    )

    # --------------------------------------------------
    # OUTPUT CONSTRUCTION
    # --------------------------------------------------

    future_index = pd.date_range(
        start   = y.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    hist_block = pd.DataFrame(
        {
            "date":      y.index,
            "actual":    np.nan,
            "forecast":  theta_fitted,
            "ci_low":    np.nan,
            "ci_mid":    theta_fitted,
            "ci_high":   np.nan,
            "error_pct": np.nan,
        }
    )

    future_block = pd.DataFrame(
        {
            "date":      future_index,
            "actual":    np.nan,
            "forecast":  forecast_values,
            "ci_low":    ci_low,
            "ci_mid":    forecast_values,
            "ci_high":   ci_high,
            "error_pct": np.nan,
        }
    )

    # Dtype governance — ensure all numeric cols are float64
    numeric_cols = ["forecast", "ci_low", "ci_mid", "ci_high"]
    hist_block[numeric_cols]   = hist_block[numeric_cols].astype("float64")
    future_block[numeric_cols] = future_block[numeric_cols].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    if not np.isfinite(forecast_values).all():
        raise RuntimeError("Theta produced non-finite values after hardening.")

    return ForecastResult(
        model_name  = "Theta",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "method":             "additive_theta",
            "hardened_fallback":  hardened_fallback,
            "frequency":          inferred,
            "confidence_level":   confidence_level,
            "ci_method":          "residual_based_sigma_sqrt_h",
            "output_contract":         "ForecastResult",
        },
    )
