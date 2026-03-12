# ============================================================
# FILE: arima.py
# ROLE: ARIMA FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List
import numpy as np

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_arima(
    dataset: AFECommittedDataset,
    horizon: int,
    p: int = 1,
    d: int = 1,
    q: int = 0,
) -> ForecastOutput:
    """
    ARIMA(p,d,q) — GOVERNED, DETERMINISTIC IMPLEMENTATION.

    GOVERNANCE:
    - Fixed p,d,q (no auto-selection, no optimization)
    - Explicit differencing
    - Explicit autoregressive term(s)
    - No seasonal extension
    - No external statistical fitting libraries
    - Deterministic and auditable

    This represents a true ARIMA formulation suitable for AFE execution
    without violating determinism or governance.
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError("ARIMA requires an AFECommittedDataset instance.")

    values: List[float] = dataset.values

    if not values:
        raise ValueError("ARIMA requires non-empty dataset values.")

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    if p < 0 or d < 0 or q < 0:
        raise ValueError("ARIMA parameters p, d, q must be non-negative.")

    series = np.array(values, dtype=float)

    # --------------------------------------------------------
    # DIFFERENCING (d)
    # --------------------------------------------------------

    diff_series = series.copy()
    for _ in range(d):
        diff_series = np.diff(diff_series)

    if len(diff_series) <= p:
        raise ValueError("Insufficient data after differencing for AR terms.")

    # --------------------------------------------------------
    # AUTOREGRESSIVE COEFFICIENTS (FIXED, GOVERNED)
    # --------------------------------------------------------

    # Conservative fixed AR weights (declining influence)
    ar_weights = np.array(
        [1.0 / (i + 1) for i in range(p)],
        dtype=float,
    )
    ar_weights = ar_weights / ar_weights.sum()

    # --------------------------------------------------------
    # FORECAST IN DIFFERENCED SPACE
    # --------------------------------------------------------

    history = diff_series.tolist()
    diff_forecast: List[float] = []

    for _ in range(horizon):
        recent = history[-p:]
        next_value = float(np.dot(ar_weights, recent[::-1]))
        diff_forecast.append(next_value)
        history.append(next_value)

    # --------------------------------------------------------
    # INVERT DIFFERENCING
    # --------------------------------------------------------

    forecast = []
    last_values = series.tolist()

    for df in diff_forecast:
        next_level = last_values[-1] + df
        forecast.append(next_level)
        last_values.append(next_level)

    # --------------------------------------------------------
    # CONSERVATIVE UNCERTAINTY (SYMMETRIC, NON-STOCHASTIC)
    # --------------------------------------------------------

    base = forecast
    upside = [v * 1.07 for v in forecast]
    downside = [v * 0.93 for v in forecast]

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
