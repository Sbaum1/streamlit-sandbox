# FILE: streamlit_sandbox/models/tbats.py
# MODEL: TBATS (State-Space Seasonal)
# STATUS: FORTUNE 100 REBUILD — STRICT INDEX / NO RUNTIME PATCHING / NO FAKE CI
# PATCH: DTYPE GOVERNANCE HARDENING (NO CONCAT WARNING)
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from tbats import TBATS

from streamlit_sandbox.models.contracts import ForecastResult


def run_tbats(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("TBATS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred. Explicit monthly index required.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"TBATS requires monthly frequency. Detected: {inferred}")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    seasonal_period = 12

    if len(y) < 2 * seasonal_period:
        raise ValueError("Minimum 2 full seasonal cycles required (>= 24 observations).")

    # --------------------------------------------------
    # MODEL FIT
    # --------------------------------------------------

    estimator = TBATS(
        seasonal_periods=[float(seasonal_period)],
        use_box_cox=False,
        use_arma_errors=False,
        n_jobs=1,
    )

    fitted_model = estimator.fit(y.values)

    hist_fitted = pd.Series(
        fitted_model.y_hat,
        index=y.index,
        dtype="float64",
    )

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

    future_index = pd.date_range(
        start=hist_fitted.index[-1],
        periods=horizon + 1,
        freq=inferred,
    )[1:]

    if not future_index.min() > hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    future_forecast = np.asarray(
        fitted_model.forecast(steps=horizon),
        dtype="float64",
    )

    if not np.isfinite(future_forecast).all():
        raise RuntimeError("Non-finite forecast values detected.")

    future_block = pd.DataFrame(
        {
            "date": future_index,
            "actual": np.nan,
            "forecast": future_forecast,
            "ci_low": np.nan,
            "ci_mid": future_forecast,
            "ci_high": np.nan,
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
        model_name="TBATS",
        forecast_df=forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "seasonal_period": seasonal_period,
            "frequency": inferred,
            "confidence_level": confidence_level,
            "box_cox": False,
            "arma_errors": False,
            "ci_method": "not_available_tbats",
            "compliance": "Fortune_100_standard",
        },
    )