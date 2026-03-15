# ==================================================
# FILE: sentinel_engine/models/stl_ets.py
# VERSION: 2.0.0
# MODEL: STL + ETS (Deterministic)
# ENGINE: Sentinel Engine v2.0.0
# STATUS: VEDUTA ENGINE / ZERO REGRESSION — REAL CI — 3B-2
# ==================================================
#
# 3B-2 CI FIX:
#   Previous: ci_low = np.nan, ci_high = np.nan (hard-coded)
#   Fixed:    Residual-based prediction intervals
#
#   Method: In-sample residuals → pooled std → horizon-scaled CI
#     residuals = y - (hist_trend + seasonal)
#     sigma     = std(residuals)
#     sigma_h   = sigma * sqrt(h)
#     ci_low    = forecast - z * sigma_h
#     ci_high   = forecast + z * sigma_h
#
#   STL+ETS is a deterministic decomposition model.
#   No native library CI is available. Residual-based
#   intervals are applied to the final combined forecast.
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sentinel_engine.models.contracts import ForecastResult


ALPHA = 0.4
BETA  = 0.1
PHI   = 0.98


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

def run_stl_ets(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("STL+ETS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"Monthly frequency required. Detected: {inferred}")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(y) < 24:
        raise ValueError("Minimum 24 observations required.")

    # --------------------------------------------------
    # STL DECOMPOSITION
    # --------------------------------------------------

    stl = STL(
        y,
        period  = 12,
        seasonal = 13,
        trend    = 13,
        robust   = True,
    )

    stl_res  = stl.fit()
    seasonal  = stl_res.seasonal
    deseasonal = y - seasonal

    # --------------------------------------------------
    # ETS ON DESEASONED SERIES
    # --------------------------------------------------

    ets_model = ExponentialSmoothing(
        deseasonal,
        trend               = "add",
        damped_trend        = True,
        seasonal            = None,
        initialization_method = "estimated",
        use_boxcox          = False,
    )

    ets = ets_model.fit(
        smoothing_level = ALPHA,
        smoothing_trend = BETA,
        damping_trend   = PHI,
        optimized       = False,
    )

    hist_trend    = ets.fittedvalues.astype("float64")
    hist_forecast = hist_trend + seasonal

    if not np.isfinite(hist_forecast).all():
        raise RuntimeError("Non-finite historical values detected.")

    # ── Residuals for CI (3B-2) ──────────────────────────────────────────────
    residuals = (y.values - hist_forecast.values).astype("float64")

    hist_block = pd.DataFrame(
        {
            "date":      hist_forecast.index,
            "actual":    np.nan,
            "forecast":  hist_forecast.values,
            "ci_low":    np.nan,
            "ci_mid":    hist_forecast.values,
            "ci_high":   np.nan,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_index = pd.date_range(
        start   = hist_forecast.index[-1],
        periods = horizon + 1,
        freq    = inferred,
    )[1:]

    future_trend   = ets.forecast(horizon).astype("float64")
    seasonal_cycle = seasonal.iloc[-12:].values.astype("float64")
    seasonal_future = np.tile(
        seasonal_cycle,
        int(np.ceil(horizon / 12))
    )[:horizon]

    base_future = future_trend.values + seasonal_future

    if not np.isfinite(base_future).all():
        raise RuntimeError("Non-finite future values detected.")

    # ── Residual CI (3B-2) ───────────────────────────────────────────────────
    ci_low, ci_high = _residual_ci(
        forecast_values  = base_future,
        residuals        = residuals,
        confidence_level = confidence_level,
    )

    future_block = pd.DataFrame(
        {
            "date":      future_index,
            "actual":    np.nan,
            "forecast":  base_future,
            "ci_low":    ci_low,
            "ci_mid":    base_future,
            "ci_high":   ci_high,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df.empty:
        raise RuntimeError("Forecast dataframe is empty.")
    if not forecast_df["forecast"].notna().all():
        raise RuntimeError("NaN forecast values detected.")
    if not np.isfinite(forecast_df["forecast"].values).all():
        raise RuntimeError("Non-finite forecast values detected.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    def _to_list(arr):
        return [float(round(x, 10)) for x in arr.astype("float64")]

    metadata = {
        "stl_period":           12,
        "stl_seasonal_window":  13,
        "stl_trend_window":     13,
        "stl_robust":           True,
        "ets_error":            "add",
        "ets_trend":            "add",
        "ets_damped":           True,
        "ets_alpha":            ALPHA,
        "ets_beta":             BETA,
        "ets_phi":              PHI,
        "optimized":            False,
        "use_boxcox":           False,
        "ci_method":            "residual_based_sigma_sqrt_h",
        "output_contract":           "ForecastResult",
        "future_trend":         _to_list(future_trend.values),
        "seasonal_future":      _to_list(seasonal_future),
        "base_future":          _to_list(base_future),
    }

    return ForecastResult(
        model_name  = "STL+ETS",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = metadata,
    )