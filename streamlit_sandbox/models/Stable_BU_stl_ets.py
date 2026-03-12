# ==================================================
# FILE: streamlit_sandbox/models/stl_ets.py
# MODEL: STL + ETS (Fully Locked Deterministic)
# STATUS: FORTUNE 100 HARDENED — SHOCK PATH VALIDATION
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from streamlit_sandbox.models.contracts import ForecastResult


# --------------------------------------------------
# LOCKED CONFIGURATION CONSTANTS (UNCHANGED)
# --------------------------------------------------

STL_PERIOD = 12
STL_SEASONAL_WINDOW = 13
STL_TREND_WINDOW = 13
STL_ROBUST = True

ALPHA = 0.25
BETA = 0.08
PHI = 0.90


def run_stl_ets(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("STL+ETS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred not in ("MS", "M"):
        raise ValueError("Monthly frequency required.")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected.")

    if len(y) < 24:
        raise ValueError("Minimum 24 observations required.")

    # --------------------------------------------------
    # STL DECOMPOSITION (LOCKED)
    # --------------------------------------------------

    stl = STL(
        y,
        period=STL_PERIOD,
        seasonal=STL_SEASONAL_WINDOW,
        trend=STL_TREND_WINDOW,
        robust=STL_ROBUST,
    ).fit()

    seasonal = stl.seasonal
    deseasonal = y - seasonal

    if seasonal.isna().any():
        raise RuntimeError("STL seasonal extraction failed.")

    # --------------------------------------------------
    # ETS (DETERMINISTIC)
    # --------------------------------------------------

    ets_model = ExponentialSmoothing(
        deseasonal,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
        use_boxcox=False,
    )

    ets = ets_model.fit(
        smoothing_level=ALPHA,
        smoothing_trend=BETA,
        damping_trend=PHI,
        optimized=False,
    )

    trend_fitted = ets.fittedvalues.astype("float64")

    if trend_fitted.isna().any():
        raise RuntimeError("ETS fitted values invalid.")

    hist_forecast = trend_fitted + seasonal

    # --------------------------------------------------
    # FUTURE FORECAST (STRICT INTEGRITY)
    # --------------------------------------------------

    future_index = pd.date_range(
        start=y.index[-1],
        periods=horizon + 1,
        freq=inferred,
    )[1:]

    if len(future_index) != horizon:
        raise RuntimeError("Future index length mismatch.")

    if not future_index.min() > y.index.max():
        raise RuntimeError("Forecast overlaps historical data.")

    future_trend = ets.forecast(horizon).astype("float64")

    if len(future_trend) != horizon:
        raise RuntimeError("Future trend length mismatch.")

    seasonal_cycle = np.asarray(
        seasonal.iloc[-STL_PERIOD:].values,
        dtype="float64",
    )

    seasonal_future = np.tile(
        seasonal_cycle,
        int(np.ceil(horizon / STL_PERIOD))
    )[:horizon]

    if len(seasonal_future) != horizon:
        raise RuntimeError("Seasonal future length mismatch.")

    base_future = future_trend.values + seasonal_future

    # --------------------------------------------------
    # STRICT FORECAST INTEGRITY ASSERTIONS
    # --------------------------------------------------

    if len(base_future) != horizon:
        raise RuntimeError("Base future length mismatch.")

    if not np.isfinite(base_future).all():
        raise RuntimeError("Non-finite forecast values detected in base_future.")

    # --------------------------------------------------
    # CONSTRUCT OUTPUT
    # --------------------------------------------------

    hist_block = pd.DataFrame(
        {
            "date": hist_forecast.index,
            "actual": pd.NA,
            "forecast": hist_forecast.values,
            "ci_low": pd.NA,
            "ci_mid": hist_forecast.values,
            "ci_high": pd.NA,
            "error_pct": pd.NA,
        }
    )

    future_block = pd.DataFrame(
        {
            "date": future_index,
            "actual": pd.NA,
            "forecast": base_future,
            "ci_low": pd.NA,
            "ci_mid": base_future,
            "ci_high": pd.NA,
            "error_pct": pd.NA,
        }
    )

    forecast_df = pd.concat(
        [hist_block, future_block],
        ignore_index=True,
    )

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # POST-CONSTRUCTION VALIDATION
    # --------------------------------------------------

    if forecast_df.empty:
        raise RuntimeError("STL+ETS produced empty forecast_df.")

    if forecast_df["forecast"].isna().any():
        raise RuntimeError("STL+ETS produced NaN forecast values.")

    if not np.isfinite(forecast_df["forecast"].values).all():
        raise RuntimeError("STL+ETS produced non-finite forecast values.")

    # --------------------------------------------------
    # RETURN (SCHEMA PRESERVED)
    # --------------------------------------------------

    return ForecastResult(
        model_name="STL+ETS",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "stl_period": STL_PERIOD,
            "stl_seasonal_window": STL_SEASONAL_WINDOW,
            "stl_trend_window": STL_TREND_WINDOW,
            "stl_robust": STL_ROBUST,
            "ets_error": "add",
            "ets_trend": "add",
            "ets_damped": True,
            "ets_alpha": ALPHA,
            "ets_beta": BETA,
            "ets_phi": PHI,
            "optimized": False,
            "use_boxcox": False,
            "shock_path_validation": True,
            "validation_len_y": len(y),
            "validation_len_seasonal": len(seasonal),
            "validation_horizon": horizon,
            "validation_len_future_trend": len(future_trend),
            "validation_len_seasonal_future": len(seasonal_future),
            "validation_len_base_future": len(base_future),
            "frequency": inferred,
            "confidence_level": confidence_level,
        },
    )