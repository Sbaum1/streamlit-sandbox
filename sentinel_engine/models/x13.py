# FILE: streamlit_sandbox/models/x13.py
# MODEL: X-13 ARIMA-SEATS (Diagnostic Only)
# STATUS: FORTUNE 100 REBUILD — STRICT INDEX / DIAGNOSTIC ONLY / NO FAKE CI
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.x13 import x13_arima_analysis

from sentinel_engine.models.contracts import ForecastResult


def run_x13(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("X-13 requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date").set_index("date")

    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred. Explicit monthly index required.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"X-13 requires monthly frequency. Detected: {inferred}")

    df = df.asfreq(inferred)

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype(float)

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    # --------------------------------------------------
    # X-13 DIAGNOSTIC EXECUTION
    # --------------------------------------------------

    try:
        res = x13_arima_analysis(y)
        trend = res.trend.dropna()

        if trend.empty:
            raise RuntimeError("X-13 produced no usable trend component.")

    except Exception as e:
        # Controlled diagnostic failure (still structurally valid output)
        empty_df = pd.DataFrame(
            columns=["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        )
        return ForecastResult(
            model_name="X-13",
            forecast_df=empty_df,
            metrics=None,
            metadata={
                "diagnostic_only": True,
                "status": "unavailable",
                "reason": str(e),
                "compliance": "Fortune_100_standard",
            },
        )

    # --------------------------------------------------
    # DIAGNOSTIC OUTPUT (NO TRUE FORECAST)
    # --------------------------------------------------

    hist_block = pd.DataFrame(
        {
            "date": trend.index,
            "actual": pd.NA,
            "forecast": trend.values,
            "ci_low": pd.NA,
            "ci_mid": trend.values,
            "ci_high": pd.NA,
            "error_pct": pd.NA,
        }
    )

    hist_block = hist_block.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name="X-13",
        forecast_df=hist_block[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics=None,
        metadata={
            "diagnostic_only": True,
            "role": "seasonal_adjustment_authority",
            "frequency": inferred,
            "ci_method": "not_applicable",
            "compliance": "Fortune_100_standard",
        },
    )
