# ============================================================
# FILE: ets.py
# ROLE: ETS (ERROR–TREND–SEASONALITY) FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List, Optional
import math

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def _initial_trend(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return values[-1] - values[-2]


def _initial_seasonality(values: List[float], season_length: int) -> List[float]:
    seasonals = [0.0] * season_length
    if len(values) < season_length * 2:
        return seasonals

    season_averages = []
    n_seasons = len(values) // season_length
    for j in range(n_seasons):
        start = j * season_length
        season_averages.append(
            sum(values[start:start + season_length]) / season_length
        )

    for i in range(season_length):
        seasonals[i] = sum(
            values[j * season_length + i] - season_averages[j]
            for j in range(n_seasons)
        ) / n_seasons

    return seasonals


def run_ets(
    dataset: AFECommittedDataset,
    horizon: int,
    season_length: Optional[int] = None,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> ForecastOutput:
    """
    ETS (Additive Error, Additive Trend, Additive Seasonality).

    GOVERNANCE:
    - Deterministic
    - No auto-optimization
    - Fixed smoothing parameters
    - Explicit, visible assumptions
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError("ETS requires an AFECommittedDataset instance.")

    values: List[float] = dataset.values

    if not values:
        raise ValueError("ETS requires non-empty dataset values.")

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    level = values[0]
    trend = _initial_trend(values)
    seasonals = (
        _initial_seasonality(values, season_length)
        if season_length
        else None
    )

    for i, value in enumerate(values):
        if seasonals:
            seasonal = seasonals[i % season_length]
            last_level = level
            level = alpha * (value - seasonal) + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonals[i % season_length] = (
                gamma * (value - level) + (1 - gamma) * seasonal
            )
        else:
            last_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend

    forecast: List[float] = []
    for h in range(1, horizon + 1):
        if seasonals:
            seasonal = seasonals[(len(values) + h - 1) % season_length]
            forecast.append(level + h * trend + seasonal)
        else:
            forecast.append(level + h * trend)

    # Conservative symmetric uncertainty (no stochastic simulation)
    base = forecast
    upside = [v * 1.05 for v in forecast]
    downside = [v * 0.95 for v in forecast]

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
