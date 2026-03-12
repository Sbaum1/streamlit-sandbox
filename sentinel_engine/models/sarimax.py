# ==================================================
# FILE: streamlit_sandbox/models/sarimax.py
# MODEL: SARIMAX (Fixed Seasonal Order)
# STATUS: FORTUNE 100 HARDENED — NO AUTO SELECTION
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sentinel_engine.models.contracts import ForecastResult


ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 12)
MAX_ITER = 200


def run_sarimax(df: pd.DataFrame, horizon: int, confidence_level: float) -> ForecastResult:

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred not in ("MS", "M"):
        raise ValueError("SARIMAX requires monthly frequency.")

    df = df.asfreq(inferred)

    y = df["value"].astype("float64")

    model = SARIMAX(
        y,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    res = model.fit(maxiter=MAX_ITER, disp=False)

    hist_fitted = res.fittedvalues.astype("float64")

    forecast_res = res.get_forecast(steps=horizon)
    future_mean = forecast_res.predicted_mean.astype("float64")
    ci = forecast_res.conf_int(alpha=1.0 - confidence_level).astype("float64")

    hist_block = pd.DataFrame(
        {
            "date": hist_fitted.index,
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
            "date": future_mean.index,
            "actual": pd.NA,
            "forecast": future_mean.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_mid": future_mean.values,
            "ci_high": ci.iloc[:, 1].values,
            "error_pct": pd.NA,
        }
    )

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name="SARIMAX",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "order": ORDER,
            "seasonal_order": SEASONAL_ORDER,
            "frequency": inferred,
        },
    )
