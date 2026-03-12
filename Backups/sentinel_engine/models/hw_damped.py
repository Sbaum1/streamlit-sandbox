# ==================================================
# FILE: sentinel_engine/models/hw_damped.py
# VERSION: 2.0.0
# MODEL: HOLT-WINTERS DAMPED TREND
# ENGINE: Sentinel Engine v2.0.0
# TIER: essentials (minimum)
# STATUS: FORTUNE 100 REBUILD — PHASE 3C
# ==================================================
#
# PURPOSE:
#   Holt-Winters Exponential Smoothing with damped trend.
#   Prevents over-extrapolation on short-horizon volatile series.
#   Outperforms standard ETS when trend is present but expected
#   to flatten — common in sales, supply chain, and revenue data.
#
# DAMPING:
#   damping_trend=True with phi parameter (0 < phi < 1).
#   phi < 1 damps the trend toward a flat forecast over the horizon.
#   Auto-fit: statsmodels selects optimal phi via MLE.
#
# CI METHOD:
#   Residual-based: sigma * sqrt(h) horizon scaling.
#   Consistent with ETS, Theta, STL+ETS CI methodology.
#
# DIFFERENCES FROM ETS (ets.py):
#   - damped_trend=True always
#   - trend="add" always (damped multiplicative is rarely stable)
#   - Separate file to allow independent MASE certification
#   - min_tier = essentials (ETS is also essentials)
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sentinel_engine.models.contracts import ForecastResult

# --------------------------------------------------
# Z-SCORE MAP FOR CI CONSTRUCTION
# --------------------------------------------------
_Z = {
    0.50: 0.674,
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

def _get_z(confidence_level: float) -> float:
    if confidence_level in _Z:
        return _Z[confidence_level]
    # Linear interpolation fallback
    levels = sorted(_Z.keys())
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        if lo <= confidence_level <= hi:
            t = (confidence_level - lo) / (hi - lo)
            return _Z[lo] + t * (_Z[hi] - _Z[lo])
    return 1.960  # default to 95%


# ==================================================
# MODEL RUNNER
# ==================================================

def run_hw_damped(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("HW Damped requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")

    if df["value"].isna().any():
        raise ValueError("Missing values detected in input series.")

    y = df["value"].astype("float64").values

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(df) < 12:
        raise ValueError("Minimum 12 observations required.")

    # --------------------------------------------------
    # SEASON LENGTH
    # --------------------------------------------------

    _season_map = {
        "MS": 12, "M": 12,
        "QS": 4,  "Q": 4,
        "W":  52, "W-SUN": 52, "W-MON": 52,
        "A":  1,  "AS": 1,
        "D":  7,
    }
    season_len = _season_map.get(inferred, 12)

    # Need at least 2 full seasonal cycles for seasonal model
    use_seasonal = len(y) >= 2 * season_len and season_len > 1

    # --------------------------------------------------
    # FIT DAMPED HOLT-WINTERS MODEL
    # --------------------------------------------------

    try:
        model = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="add" if use_seasonal else None,
            seasonal_periods=season_len if use_seasonal else None,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, remove_bias=True)
    except Exception as e:
        raise RuntimeError(f"HW Damped model fit failed: {e}") from e

    # --------------------------------------------------
    # RESIDUALS AND SIGMA
    # --------------------------------------------------

    residuals = fitted.resid
    finite_resid = residuals[np.isfinite(residuals)]
    if len(finite_resid) < 2:
        raise RuntimeError("Insufficient residuals for CI computation.")

    sigma = float(np.std(finite_resid, ddof=1))
    z     = _get_z(confidence_level)

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    fitted_values = fitted.fittedvalues.astype("float64")

    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  fitted_values,
        "ci_low":    np.nan,
        "ci_mid":    fitted_values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_forecast = fitted.forecast(horizon).astype("float64")

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite forecast values detected.")

    # Build future dates
    last_date  = df["date"].iloc[-1]
    freq_alias = inferred if inferred else "MS"
    future_idx = pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq=freq_alias
    )[1:]

    if len(future_idx) != horizon:
        raise RuntimeError(f"Future date index length mismatch: {len(future_idx)} vs {horizon}.")

    # Residual-based CI with horizon scaling
    h_arr  = np.arange(1, horizon + 1, dtype="float64")
    ci_low  = future_forecast - z * sigma * np.sqrt(h_arr)
    ci_high = future_forecast + z * sigma * np.sqrt(h_arr)

    future_block = pd.DataFrame({
        "date":      future_idx,
        "actual":    np.nan,
        "forecast":  future_forecast,
        "ci_low":    ci_low.astype("float64"),
        "ci_mid":    future_forecast,
        "ci_high":   ci_high.astype("float64"),
        "error_pct": np.nan,
    })

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

    # Validate CI integrity on future block
    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI values in future forecast output.")
    if (future_rows["ci_low"] >= future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI detected in future forecast output.")

    return ForecastResult(
        model_name  = "HW_Damped",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "trend":             "additive_damped",
            "seasonal":          "additive" if use_seasonal else "none",
            "seasonal_periods":  season_len if use_seasonal else None,
            "damped_trend":      True,
            "phi":               float(fitted.params.get("damping_trend", np.nan)),
            "ci_method":         "residual_based_sigma_sqrt_h",
            "sigma":             round(sigma, 6),
            "z_score":           z,
            "frequency":         inferred,
            "confidence_level":  confidence_level,
            "min_tier":          "essentials",
            "compliance":        "Fortune_100_standard",
        },
    )