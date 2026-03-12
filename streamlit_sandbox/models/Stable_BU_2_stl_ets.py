# ==================================================
# FILE: streamlit_sandbox/models/stl_ets.py
# MODEL: STL + ETS (Deterministic / Diagnostic Instrumented)
# STANDARD: FORTUNE 100 / ZERO REGRESSION
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from streamlit_sandbox.models.contracts import ForecastResult


# --------------------------------------------------
# DETERMINISTIC SMOOTHING CONSTANTS
# --------------------------------------------------

ALPHA = 0.4
BETA = 0.1
PHI = 0.98


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
    # STL (LOCKED CONFIGURATION)
    # --------------------------------------------------

    stl = STL(
        y,
        period=12,
        seasonal=13,
        trend=13,
        robust=True,
    )

    stl_res = stl.fit()

    seasonal = stl_res.seasonal
    trend_component = y - seasonal

    # --------------------------------------------------
    # ETS (DETERMINISTIC / BOXCOX DECLARED ONCE)
    # --------------------------------------------------

    ets_model = ExponentialSmoothing(
        trend_component,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
        use_boxcox=False,
    )

    ets_fit = ets_model.fit(
        smoothing_level=ALPHA,
        smoothing_trend=BETA,
        damping_trend=PHI,
        optimized=False,
    )

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    hist_trend = ets_fit.fittedvalues.astype("float64")
    hist_forecast = hist_trend + seasonal

    if not np.isfinite(hist_forecast).all():
        raise RuntimeError("Non-finite historical values detected.")

    hist_block = pd.DataFrame(
        {
            "date": hist_forecast.index,
            "actual": np.nan,
            "forecast": hist_forecast.values,
            "ci_low": np.nan,
            "ci_mid": hist_forecast.values,
            "ci_high": np.nan,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future_index = pd.date_range(
        start=hist_forecast.index[-1],
        periods=horizon + 1,
        freq=inferred,
    )[1:]

    if len(future_index) != horizon:
        raise RuntimeError("Future index length mismatch.")

    if future_index.min() <= hist_forecast.index.max():
        raise RuntimeError("Future dates overlap historical data.")

    future_trend = ets_fit.forecast(horizon).astype("float64")

    if len(future_trend) != horizon:
        raise RuntimeError("Future trend length mismatch.")

    seasonal_cycle = seasonal.iloc[-12:].values.astype("float64")
    seasonal_future = np.tile(
        seasonal_cycle,
        int(np.ceil(horizon / 12))
    )[:horizon]

    if len(seasonal_future) != horizon:
        raise RuntimeError("Seasonal reconstruction length mismatch.")

    base_future = future_trend.values + seasonal_future

    if not np.isfinite(base_future).all():
        raise RuntimeError("Non-finite future values detected.")

    future_block = pd.DataFrame(
        {
            "date": future_index,
            "actual": np.nan,
            "forecast": base_future,
            "ci_low": np.nan,
            "ci_mid": base_future,
            "ci_high": np.nan,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FINAL ASSEMBLY
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df.empty:
        raise RuntimeError("Forecast dataframe is empty.")

    if not forecast_df["forecast"].notna().all():
        raise RuntimeError("NaN forecast values detected.")

    if not np.isfinite(forecast_df["forecast"].values).all():
        raise RuntimeError("Non-finite forecast values detected.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # DIAGNOSTIC METADATA (NO LOGIC CHANGES)
    # --------------------------------------------------

    def _round_list(arr):
        return [float(round(x, 10)) for x in arr.astype("float64")]

    base_future_arr = base_future.astype("float64")
    future_trend_arr = future_trend.values.astype("float64")
    seasonal_future_arr = seasonal_future.astype("float64")
    deseasonal_last_12_arr = trend_component.iloc[-12:].values.astype("float64")
    seasonal_last_12_arr = seasonal.iloc[-12:].values.astype("float64")

    var_future_total = float(round(np.var(base_future_arr), 10))
    var_future_trend = float(round(np.var(future_trend_arr), 10))
    var_future_seasonal = float(round(np.var(seasonal_future_arr), 10))

    amplification_ratio_trend = (
        float(round(var_future_trend / var_future_total, 10))
        if var_future_total > 0 else None
    )

    amplification_ratio_seasonal = (
        float(round(var_future_seasonal / var_future_total, 10))
        if var_future_total > 0 else None
    )

    metadata = {
        "stl_period": 12,
        "stl_seasonal_window": 13,
        "stl_trend_window": 13,
        "stl_robust": True,
        "ets_error": "add",
        "ets_trend": "add",
        "ets_damped": True,
        "ets_alpha": ALPHA,
        "ets_beta": BETA,
        "ets_phi": PHI,
        "optimized": False,
        "use_boxcox": False,
        "base_future": _round_list(base_future_arr),
        "future_trend": _round_list(future_trend_arr),
        "seasonal_future": _round_list(seasonal_future_arr),
        "deseasonal_last_12": _round_list(deseasonal_last_12_arr),
        "seasonal_last_12": _round_list(seasonal_last_12_arr),
        "var_future_total": var_future_total,
        "var_future_trend": var_future_trend,
        "var_future_seasonal": var_future_seasonal,
        "amplification_ratio_trend": amplification_ratio_trend,
        "amplification_ratio_seasonal": amplification_ratio_seasonal,
    }

    return ForecastResult(
        model_name="STL+ETS",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata=metadata,
    )