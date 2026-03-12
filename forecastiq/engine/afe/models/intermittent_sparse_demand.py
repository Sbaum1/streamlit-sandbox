# ============================================================
# FILE: intermittent_sparse_demand.py
# ROLE: INTERMITTENT / SPARSE DEMAND (CROSTON-STYLE)
# STATUS: AFE FORECAST MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List
import math

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_intermittent_sparse_demand(
    dataset: AFECommittedDataset,
    horizon: int,
    alpha: float = 0.1,
) -> ForecastOutput:
    """
    Croston-style intermittent demand forecasting.

    GOVERNANCE (LOCKED):
    - Deterministic
    - Fixed smoothing parameter (alpha)
    - No optimization or auto-tuning
    - Designed ONLY for sparse / zero-inflated demand
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError(
            "Intermittent demand model requires an AFECommittedDataset instance."
        )

    values: List[float] = dataset.values

    if not values:
        raise ValueError(
            "Intermittent demand model requires non-empty dataset values."
        )

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be in (0, 1].")

    # --------------------------------------------------------
    # CROSTON DECOMPOSITION
    # --------------------------------------------------------

    demand = None
    interval = None
    last_demand_index = None

    for i, value in enumerate(values):
        if value > 0:
            if demand is None:
                demand = value
                interval = 1
            else:
                interval = i - last_demand_index
                demand = alpha * value + (1 - alpha) * demand
            last_demand_index = i

    if demand is None or interval is None:
        # No non-zero demand observed → forecast zeros
        forecast = [0.0] * horizon
    else:
        rate = demand / interval
        forecast = [rate] * horizon

    # --------------------------------------------------------
    # CONSERVATIVE UNCERTAINTY (ASYMMETRIC, SPARSE-AWARE)
    # --------------------------------------------------------

    base = forecast
    upside = [v * 1.3 for v in forecast]
    downside = [max(0.0, v * 0.7) for v in forecast]

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
