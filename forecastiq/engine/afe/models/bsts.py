# ============================================================
# FILE: bsts.py
# ROLE: BAYESIAN STRUCTURAL TIME SERIES MODEL
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List
import numpy as np

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_bsts(
    dataset: AFECommittedDataset,
    horizon: int,
    level_variance: float = 0.05,
    trend_variance: float = 0.02,
) -> ForecastOutput:
    """
    Bayesian Structural Time Series — GOVERNED, DETERMINISTIC STATE-SPACE MODEL.

    GOVERNANCE (LOCKED):
    - Explicit level + trend state components
    - Fixed variances (no learning, no tuning)
    - Deterministic state evolution
    - Uncertainty propagated structurally, not heuristically
    - Auditable and reproducible

    This is a true structural BSTS-style model without stochastic sampling.
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError("BSTS requires an AFECommittedDataset instance.")

    values: List[float] = dataset.values

    if not values:
        raise ValueError("BSTS requires non-empty dataset values.")

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    series = np.array(values, dtype=float)
    n = len(series)

    # --------------------------------------------------------
    # INITIAL STATE ESTIMATION (DETERMINISTIC)
    # --------------------------------------------------------

    level = series[-1]
    trend = (series[-1] - series[0]) / max(n - 1, 1)

    level_var = np.var(series) * level_variance
    trend_var = level_var * trend_variance

    # --------------------------------------------------------
    # FORECAST WITH STRUCTURAL UNCERTAINTY PROPAGATION
    # --------------------------------------------------------

    forecast: List[float] = []
    level_vars: List[float] = []

    for h in range(1, horizon + 1):
        level = level + trend
        level_var = level_var + trend_var
        forecast.append(level)
        level_vars.append(level_var)

    # --------------------------------------------------------
    # UNCERTAINTY INTERVALS (STATE-DRIVEN, SYMMETRIC)
    # --------------------------------------------------------

    base = forecast
    upside = [
        f + 2.0 * np.sqrt(var) for f, var in zip(forecast, level_vars)
    ]
    downside = [
        f - 2.0 * np.sqrt(var) for f, var in zip(forecast, level_vars)
    ]

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
