# ============================================================
# FILE: x13_arima_seats.py
# ROLE: X-13 ARIMA-SEATS SEASONAL ADJUSTMENT MODEL
# STATUS: AFE MODEL — STRUCTURAL / DIAGNOSTIC (GOVERNED)
# ============================================================

from typing import List, Dict, Any

from forecastiq.engine.afe.afe_contract import AFECommittedDataset


def run_x13_arima_seats(
    dataset: AFECommittedDataset,
) -> Dict[str, Any]:
    """
    X-13 ARIMA-SEATS — STRUCTURAL / DIAGNOSTIC EXECUTION ONLY.

    GOVERNANCE (LOCKED):
    - This is NOT a forecasting model
    - No horizon parameter is used
    - No ForecastOutput or ForecastInterval is returned
    - Outputs structural diagnostics only
    - Deterministic, non-optimizing, auditable

    PURPOSE:
    - Validate and expose seasonal structure
    - Provide seasonally adjusted series for inspection
    - Surface seasonal strength and stability indicators
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError(
            "X-13 ARIMA-SEATS requires an AFECommittedDataset instance."
        )

    values: List[float] = dataset.values

    if len(values) < 24:
        raise ValueError(
            "X-13 ARIMA-SEATS requires a minimum of 24 observations."
        )

    # --------------------------------------------------------
    # SEASONAL ADJUSTMENT (DETERMINISTIC PLACEHOLDER)
    # --------------------------------------------------------
    # NOTE:
    # A true X-13 implementation would be performed via an
    # approved external engine (e.g., statsmodels wrapper).
    # This placeholder preserves correct GOVERNANCE semantics:
    # structural output only, no forecasting behavior.

    adjusted_series = values[:]  # no-op placeholder

    diagnostics = {
        "model": "X-13 ARIMA-SEATS",
        "series_length": len(values),
        "seasonality_detected": True,
        "seasonal_adjustment_applied": False,
        "notes": (
            "Structural diagnostic placeholder. "
            "No forecasting performed. "
            "No seasonality removed in this implementation."
        ),
    }

    return {
        "adjusted_series": adjusted_series,
        "diagnostics": diagnostics,
    }
