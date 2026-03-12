# ==================================================
# FILE: streamlit_sandbox/models/theta.py
# MODEL: THETA (Additive / Deterministic)
# STATUS: FORTUNE 100 HARDENED — NON-FINITE PROTECTION
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from streamlit_sandbox.models.contracts import ForecastResult


EPSILON = 1e-8
CLIP_LIMIT = 1e12


def run_theta(
    df: pd.DataFrame,
    horizon: int,
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

    t = np.arange(len(y), dtype="float64")
    y_values = y.values.astype("float64")

    # Linear regression component (Theta=2 equivalent)
    slope = np.sum((t - t.mean()) * (y_values - y_values.mean())) / (
        np.sum((t - t.mean()) ** 2) + EPSILON
    )
    intercept = y_values.mean() - slope * t.mean()

    trend_component = intercept + slope * t

    # SES component (Theta=0 equivalent)
    ses_model = SimpleExpSmoothing(y_values, initialization_method="estimated")
    ses_fit = ses_model.fit(optimized=False, smoothing_level=0.2)

    ses_fitted = ses_fit.fittedvalues.astype("float64")

    # Combine components
    theta_fitted = 0.5 * (trend_component + ses_fitted)

    theta_fitted = np.clip(theta_fitted, -CLIP_LIMIT, CLIP_LIMIT)

    if not np.isfinite(theta_fitted).all():
        raise RuntimeError("Theta produced non-finite fitted values.")

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_t = np.arange(len(y), len(y) + horizon, dtype="float64")

    trend_future = intercept + slope * future_t

    ses_last = ses_fitted[-1]
    if not np.isfinite(ses_last):
        ses_last = y_values[-1]

    ses_future = np.full(horizon, ses_last, dtype="float64")

    forecast_values = 0.5 * (trend_future + ses_future)

    forecast_values = np.clip(forecast_values, -CLIP_LIMIT, CLIP_LIMIT)

    # Hard non-finite guard
    if not np.isfinite(forecast_values).all():
        # Deterministic fallback: linear extrapolation only
        fallback_forecast = intercept + slope * future_t
        fallback_forecast = np.clip(fallback_forecast, -CLIP_LIMIT, CLIP_LIMIT)

        if not np.isfinite(fallback_forecast).all():
            raise RuntimeError("Theta produced non-finite values after hardening")

        forecast_values = fallback_forecast
        hardened_fallback = True
    else:
        hardened_fallback = False

    # --------------------------------------------------
    # OUTPUT CONSTRUCTION
    # --------------------------------------------------

    future_index = pd.date_range(
        start=y.index[-1],
        periods=horizon + 1,
        freq=inferred,
    )[1:]

    hist_block = pd.DataFrame(
        {
            "date": y.index,
            "actual": pd.NA,
            "forecast": theta_fitted,
            "ci_low": pd.NA,
            "ci_mid": theta_fitted,
            "ci_high": pd.NA,
            "error_pct": pd.NA,
        }
    )

    future_block = pd.DataFrame(
        {
            "date": future_index,
            "actual": pd.NA,
            "forecast": forecast_values,
            "ci_low": pd.NA,
            "ci_mid": forecast_values,
            "ci_high": pd.NA,
            "error_pct": pd.NA,
        }
    )

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    if not np.isfinite(forecast_values).all():
        raise RuntimeError("Theta produced non-finite values after hardening")

    return ForecastResult(
        model_name="Theta",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "method": "additive_theta",
            "hardened_fallback": hardened_fallback,
            "frequency": inferred,
            "confidence_level": confidence_level,
            "compliance": "Fortune_100_standard",
        },
    )