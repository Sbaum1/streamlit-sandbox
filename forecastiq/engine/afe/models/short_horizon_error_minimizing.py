# ============================================================
# FILE: short_horizon_error_minimizing.py
# ROLE: SHORT-HORIZON ERROR-MINIMIZING FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List
import numpy as np

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_short_horizon_error_minimizing(
    dataset: AFECommittedDataset,
    horizon: int,
    window: int = 5,
) -> ForecastOutput:
    """
    Short-Horizon Error-Minimizing Model — GOVERNED, DETERMINISTIC.

    GOVERNANCE (LOCKED):
    - Fixed recent window (no optimization)
    - Emphasizes most recent dynamics
    - Ignores long-term structure intentionally
    - Deterministic and auditable
    - Designed ONLY for short-horizon behavior
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError(
            "Short-Horizon Error-Minimizing model requires an AFECommittedDataset instance."
        )

    values: List[float] = dataset.values

    if not values:
        raise ValueError(
            "Short-Horizon Error-Minimizing model requires non-empty dataset values."
        )

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    if window <= 1:
        raise ValueError("Window size must be greater than 1.")

    series = np.array(values, dtype=float)

    if len(series) < window:
        raise ValueError(
            "Insufficient data length for the configured short-horizon window."
        )

    # --------------------------------------------------------
    # RECENT-WINDOW ERROR MINIMIZATION LOGIC
    # --------------------------------------------------------
    # Use recent first differences to estimate near-term movement

    recent = series[-window:]
    diffs = np.diff(recent)

    # Weighted emphasis on most recent movements
    weights = np.linspace(1.0, 2.0, num=len(diffs))
    weights = weights / weights.sum()

    avg_diff = float(np.dot(weights, diffs))

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    forecast: List[float] = []
    last_value = series[-1]

    for h in range(1, horizon + 1):
        next_value = last_value + avg_diff
        forecast.append(next_value)
        last_value = next_value

    # --------------------------------------------------------
    # CONSERVATIVE UNCERTAINTY (NARROW, SHORT-HORIZON)
    # --------------------------------------------------------

    base = forecast
    upside = [v * 1.04 for v in forecast]
    downside = [v * 0.96 for v in forecast]

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
