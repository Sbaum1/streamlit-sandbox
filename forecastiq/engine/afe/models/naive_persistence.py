# ============================================================
# FILE: naive_persistence.py
# ROLE: NAÏVE / PERSISTENCE FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED
# ============================================================

from typing import List

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_naive_persistence(
    dataset: AFECommittedDataset,
    horizon: int,
) -> ForecastOutput:
    """
    Naïve persistence forecast.

    Governance rules:
    - Operates ONLY on committed dataset values
    - No access to intelligence, suitability, or authorization
    - No transformation, tuning, smoothing, or optimization
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError(
            "Naïve persistence requires an AFECommittedDataset instance."
        )

    if not hasattr(dataset, "values"):
        raise AttributeError(
            "AFECommittedDataset must expose a `values` attribute."
        )

    values: List[float] = dataset.values

    if not values:
        raise ValueError(
            "Naïve persistence requires non-empty dataset values."
        )

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be a positive integer."
        )

    last_value = values[-1]
    forecast = [last_value] * horizon

    intervals = ForecastInterval(
        base=forecast,
        upside=forecast,
        downside=forecast,
    )

    return ForecastOutput(
        horizon=horizon,
        point_forecast=forecast,
        intervals=intervals,
    )
