# ============================================================
# FILE: sarima.py
# ROLE: SARIMA FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List
import numpy as np

from forecastiq.engine.afe.afe_contract import (
    AFECommittedDataset,
    AFEDatasetIntelligence,
)
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_sarima(
    dataset: AFECommittedDataset,
    intelligence: AFEDatasetIntelligence,
    horizon: int,
    seasonal_period: int,
    p: int = 1,
    d: int = 1,
    q: int = 0,
    P: int = 1,
    D: int = 1,
    Q: int = 0,
) -> ForecastOutput:
    """
    SARIMA — GOVERNED, DETERMINISTIC IMPLEMENTATION.

    GOVERNANCE (LOCKED):
    - Fixed (p,d,q)(P,D,Q,s) parameters
    - Explicit seasonal differencing
    - No parameter search or optimization
    - Deterministic AR propagation only
    """

    # --------------------------------------------------------
    # GOVERNANCE GATES
    # --------------------------------------------------------

    if not intelligence.seasonality_detected:
        raise NotImplementedError(
            "SARIMA blocked: seasonality not confirmed by Dataset Intelligence."
        )

    if seasonal_period not in intelligence.dominant_periods:
        raise NotImplementedError(
            f"SARIMA blocked: seasonal_period={seasonal_period} "
            f"not in dominant periods {intelligence.dominant_periods}."
        )

    values: List[float] = dataset.values

    if len(values) < seasonal_period * (D + 1):
        raise NotImplementedError(
            "SARIMA blocked: insufficient history for seasonal differencing."
        )

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    series = np.array(values, dtype=float)

    # Non-seasonal differencing (d)
    for _ in range(d):
        series = np.diff(series)

    # Seasonal differencing (D)
    for _ in range(D):
        series = series[seasonal_period:] - series[:-seasonal_period]

    if len(series) <= max(p, P):
        raise ValueError("Insufficient data after differencing for SAR terms.")

    # --------------------------------------------------------
    # FIXED AR STRUCTURE (NO MA TERMS)
    # --------------------------------------------------------

    # Estimate AR coefficient deterministically
    ar_coeff = np.corrcoef(series[:-1], series[1:])[0, 1]
    ar_coeff = 0.0 if np.isnan(ar_coeff) else ar_coeff

    # Seasonal AR coefficient
    sar_coeff = ar_coeff * 0.5

    last_values = series[-max(seasonal_period, 1):].tolist()

    # --------------------------------------------------------
    # FORECAST GENERATION
    # --------------------------------------------------------

    forecast_diffs = []
    prev = series[-1]

    for h in range(horizon):
        ar_term = ar_coeff * prev
        sar_term = sar_coeff * last_values[-seasonal_period] if len(last_values) >= seasonal_period else 0.0
        next_diff = ar_term + sar_term
        forecast_diffs.append(next_diff)
        prev = next_diff
        last_values.append(next_diff)

    # --------------------------------------------------------
    # INVERT DIFFERENCING
    # --------------------------------------------------------

    level = values[-1]
    forecast = []

    for diff in forecast_diffs:
        level = level + diff
        forecast.append(level)

    # --------------------------------------------------------
    # CONSERVATIVE UNCERTAINTY
    # --------------------------------------------------------

    std = np.std(forecast_diffs) if forecast_diffs else 0.0

    intervals = ForecastInterval(
        base=forecast,
        upside=[v + 2 * std for v in forecast],
        downside=[v - 2 * std for v in forecast],
    )

    return ForecastOutput(
        horizon=horizon,
        point_forecast=forecast,
        intervals=intervals,
    )
