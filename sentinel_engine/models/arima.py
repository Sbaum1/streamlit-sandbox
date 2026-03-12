# ==================================================
# FILE: streamlit_sandbox/models/arima.py
# MODEL: ARIMA (Deterministic Fixed Variants)
# STATUS: FORTUNE 100 HARDENED — NO AUTO ORDER
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from sentinel_engine.models.contracts import ForecastResult


ARIMA_VARIANT = "011"  # "baseline" | "011" | "100"
MAX_ITER = 200

VARIANT_MAP = {
    "baseline": (1, 1, 1),
    "011": (0, 1, 1),
    "100": (1, 0, 0),
}


def run_arima(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    if ARIMA_VARIANT not in VARIANT_MAP:
        raise RuntimeError("Invalid ARIMA_VARIANT configuration.")

    ORDER = VARIANT_MAP[ARIMA_VARIANT]

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("ARIMA requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred not in ("MS", "M"):
        raise ValueError("ARIMA requires monthly frequency.")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    try:
        model = ARIMA(
            y,
            order=ORDER,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res = model.fit(method_kwargs={"maxiter": MAX_ITER, "disp": 0})
    except Exception as e:
        raise RuntimeError(f"ARIMA fit failed: {str(e)}")

    hist_fitted = res.fittedvalues.astype("float64")

    forecast_res = res.get_forecast(steps=horizon)
    future_mean = forecast_res.predicted_mean.astype("float64")
    ci = forecast_res.conf_int(alpha=1.0 - confidence_level).astype("float64")

    if len(future_mean) != horizon:
        raise RuntimeError("Forecast length mismatch.")

    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values detected.")

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
        model_name="ARIMA",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "order": ORDER,
            "variant": ARIMA_VARIANT,
            "auto_order_disabled": True,
            "frequency": inferred,
            "confidence_level": confidence_level,
        },
    )
