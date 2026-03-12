# FILE: streamlit_sandbox/models/bsts.py
# MODEL: BSTS (Unobserved Components � Local Level + Trend + Seasonal 12)
# STATUS: VEDUTA ENGINE � STRICT INDEX / HONEST CI / NO INTERNAL METRICS
# PATCH: DTYPE GOVERNANCE HARDENING (NO CONCAT WARNING)
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from sentinel_engine.models.contracts import ForecastResult


def run_bsts(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("BSTS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred. Explicit monthly index required.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"BSTS requires monthly frequency. Detected: {inferred}")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(y) < 24:
        raise ValueError("Minimum 24 observations required (2 seasonal cycles).")

    # --------------------------------------------------
    # STATE-SPACE MODEL
    # --------------------------------------------------

    model = UnobservedComponents(
        y,
        level="local linear trend",
        seasonal=12,
    )

    res = model.fit(disp=False)

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    hist_fitted = res.fittedvalues.astype("float64")

    if hist_fitted.isna().any():
        raise RuntimeError("NaN in fitted values.")

    hist_block = pd.DataFrame(
        {
            "date": hist_fitted.index,
            "actual": np.nan,
            "forecast": hist_fitted.values,
            "ci_low": np.nan,
            "ci_mid": hist_fitted.values,
            "ci_high": np.nan,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    forecast_res = res.get_forecast(steps=horizon)
    future_mean = forecast_res.predicted_mean.astype("float64")
    ci = forecast_res.conf_int(alpha=1.0 - confidence_level).astype("float64")

    if future_mean.index.min() <= hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    if ci.isna().any().any():
        raise RuntimeError("Invalid confidence intervals detected.")

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values detected.")

    if (ci.iloc[:, 0] > ci.iloc[:, 1]).any():
        raise RuntimeError("CI bounds inverted.")

    future_block = pd.DataFrame(
        {
            "date": future_mean.index,
            "actual": np.nan,
            "forecast": future_mean.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_mid": future_mean.values,
            "ci_high": ci.iloc[:, 1].values,
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
        model_name="BSTS",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "structure": "local_level + trend + seasonal(12)",
            "engine": "state_space_kalman",
            "bayesian_style": True,
            "frequency": inferred,
            "confidence_level": confidence_level,
            "ci_method": "state_space",
            "output_contract": "ForecastResult",
        },
    )
