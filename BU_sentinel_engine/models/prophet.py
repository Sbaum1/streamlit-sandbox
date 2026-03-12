# FILE: streamlit_sandbox/models/prophet_model.py
# MODEL: PROPHET (Additive, Deterministic, Honest CI)
# STATUS: FORTUNE 100 REBUILD — STRICT INDEX / NO FABRICATED CI / NO METRICS
# PATCH: DTYPE GOVERNANCE HARDENING (NO CONCAT WARNING)
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet

from streamlit_sandbox.models.contracts import ForecastResult


def run_prophet(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Prophet requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date")

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred. Explicit monthly index required.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"Prophet requires monthly frequency. Detected: {inferred}")

    if df["value"].isna().any():
        raise ValueError("Missing values detected in input series.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(df) < 24:
        raise ValueError("Minimum 24 observations required.")

    # --------------------------------------------------
    # PROPHET MODEL
    # --------------------------------------------------

    prophet_df = df.rename(columns={"date": "ds", "value": "y"})[["ds", "y"]]

    model = Prophet(
        interval_width=confidence_level,
        uncertainty_samples=1000,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
    )

    model.fit(prophet_df)

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    fitted_hist = model.predict(prophet_df)

    if fitted_hist[["yhat", "yhat_lower", "yhat_upper"]].isna().any().any():
        raise RuntimeError("Invalid fitted output detected.")

    hist_block = pd.DataFrame(
        {
            "date": fitted_hist["ds"],
            "actual": np.nan,
            "forecast": fitted_hist["yhat"].astype("float64").values,
            "ci_low": fitted_hist["yhat_lower"].astype("float64").values,
            "ci_mid": fitted_hist["yhat"].astype("float64").values,
            "ci_high": fitted_hist["yhat_upper"].astype("float64").values,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future = model.make_future_dataframe(
        periods=horizon,
        freq=inferred,
        include_history=False,
    )

    forecast_future = model.predict(future)

    if forecast_future[["yhat", "yhat_lower", "yhat_upper"]].isna().any().any():
        raise RuntimeError("Invalid future forecast output detected.")

    if not np.isfinite(forecast_future["yhat"]).all():
        raise RuntimeError("Non-finite forecast values detected.")

    if forecast_future["ds"].min() <= df["date"].max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    future_block = pd.DataFrame(
        {
            "date": forecast_future["ds"],
            "actual": np.nan,
            "forecast": forecast_future["yhat"].astype("float64").values,
            "ci_low": forecast_future["yhat_lower"].astype("float64").values,
            "ci_mid": forecast_future["yhat"].astype("float64").values,
            "ci_high": forecast_future["yhat_upper"].astype("float64").values,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # DTYPE GOVERNANCE FIX
    # --------------------------------------------------

    numeric_cols = ["forecast", "ci_low", "ci_mid", "ci_high"]

    hist_block[numeric_cols] = hist_block[numeric_cols].astype("float64")
    future_block[numeric_cols] = future_block[numeric_cols].astype("float64")

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name="Prophet",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "trend": "piecewise_linear",
            "seasonality": "yearly",
            "changepoints": "automatic",
            "bayesian_ci": True,
            "frequency": inferred,
            "confidence_level": confidence_level,
            "compliance": "Fortune_100_standard",
        },
    )