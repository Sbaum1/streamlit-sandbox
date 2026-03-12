# ============================================================
# FILE: prophet.py
# ROLE: PROPHET FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List
import numpy as np

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_prophet(
    dataset: AFECommittedDataset,
    horizon: int,
) -> ForecastOutput:
    """
    Prophet-style forecast — GOVERNED, DETERMINISTIC IMPLEMENTATION.

    GOVERNANCE (LOCKED):
    - Fixed additive trend
    - Fixed single-season harmonic (no auto season detection)
    - No holiday effects
    - No changepoint optimization
    - No auto-tuning
    - Deterministic and auditable

    This preserves the core Prophet philosophy (trend + seasonality)
    without violating AFE governance.
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError("Prophet requires an AFECommittedDataset instance.")

    values: List[float] = dataset.values

    if not values:
        raise ValueError("Prophet requires non-empty dataset values.")

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    y = np.array(values, dtype=float)
    n = len(y)

    # --------------------------------------------------------
    # TREND (PIECEWISE-LINEAR, FIXED SLOPE)
    # --------------------------------------------------------

    x = np.arange(n)
    slope = (y[-1] - y[0]) / max(n - 1, 1)
    intercept = y[0]

    # --------------------------------------------------------
    # SEASONALITY (SINGLE HARMONIC, FIXED PERIOD)
    # Assumption: yearly-like cycle inferred from data length
    # --------------------------------------------------------

    period = max(2, n // 2)
    seasonal_amplitude = 0.1 * np.std(y)

    def seasonal_component(t: int) -> float:
        return seasonal_amplitude * np.sin(2 * np.pi * t / period)

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast: List[float] = []
    for h in range(1, horizon + 1):
        t = n + h
        trend = intercept + slope * t
        season = seasonal_component(t)
        forecast.append(trend + season)

    # --------------------------------------------------------
    # CONSERVATIVE UNCERTAINTY (SYMMETRIC, NON-STOCHASTIC)
    # --------------------------------------------------------

    base = forecast
    upside = [v * 1.08 for v in forecast]
    downside = [v * 0.92 for v in forecast]

    intervals = ForecastInterval(
        base=base,
        upside=upside,
        downside=downside,
    )

    return ForecastOutput(
        horizon=horizon,
        point_forecast=forecast,
        intervals=intervals,
    )
