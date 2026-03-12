# FILE: streamlit_sandbox/models/naive.py
# MODEL: NAIVE (Random Walk / Persistence)
# STATUS: VEDUTA ENGINE � INTERNAL METRICS / HONEST RW CI / STRICT VALIDATION
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel_engine.models.contracts import ForecastResult


def _compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    errors = actual - forecast
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.nanmean(np.abs(errors / actual)) * 100

    smape = np.mean(
        200 * abs_errors / (np.abs(actual) + np.abs(forecast))
    )

    bias = float(np.mean(errors))
    r2 = 1 - (np.sum(errors ** 2) / np.sum((actual - np.mean(actual)) ** 2))

    residual_std = float(np.std(errors, ddof=1))

    if len(errors) > 1:
        acf1 = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
    else:
        acf1 = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "Bias": bias,
        "R2_in_sample": float(r2),
        "Residual_STD": residual_std,
        "Residual_ACF1": acf1,
    }


def run_naive(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Naive model requires 'date' and 'value' columns.")

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

    y = df["value"].astype(float)

    if len(y) < 3:
        raise ValueError("Minimum 3 observations required.")

    # Historical fitted
    hist_fitted = y.shift(1)
    hist_fitted.iloc[0] = y.iloc[0]

    # Compute metrics
    metrics = _compute_metrics(
        actual=y.values,
        forecast=hist_fitted.values,
    )

    # Future forecast
    last_value = float(y.iloc[-1])
    future_index = pd.date_range(
        start=y.index[-1],
        periods=horizon + 1,
        freq=inferred,
    )[1:]

    future_forecast = np.full(horizon, last_value)

    # Random walk CI
    residuals = y - hist_fitted
    sigma = float(np.std(residuals, ddof=1))

    z_lookup = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.96}
    z = z_lookup.get(round(confidence_level, 2))
    if z is None:
        raise ValueError("Unsupported confidence level.")

    steps = np.arange(1, horizon + 1)
    interval_width = z * sigma * np.sqrt(steps)

    ci_low = future_forecast - interval_width
    ci_high = future_forecast + interval_width

    hist_block = pd.DataFrame(
        {
            "date": y.index,
            "actual": pd.NA,
            "forecast": hist_fitted.values,
            "ci_low": pd.NA,
            "ci_mid": hist_fitted.values,
            "ci_high": pd.NA,
            "error_pct": pd.NA,
        }
    )

    future_block = pd.DataFrame(
        {
            "date": future_index,
            "actual": pd.NA,
            "forecast": future_forecast,
            "ci_low": ci_low,
            "ci_mid": future_forecast,
            "ci_high": ci_high,
            "error_pct": pd.NA,
        }
    )

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    return ForecastResult(
        model_name="Naive",
        forecast_df=forecast_df,
        metrics=metrics,
        metadata={
            "model_type": "random_walk",
            "frequency": inferred,
            "confidence_level": confidence_level,
            "ci_method": "random_walk_variance",
            "output_contract": "ForecastResult",
        },
    )
