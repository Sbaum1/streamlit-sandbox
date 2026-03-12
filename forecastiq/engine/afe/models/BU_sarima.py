# ============================================================
# FILE: sarima.py
# ROLE: SARIMA FORECAST MODEL
# STATUS: AFE MODEL — GOVERNED (SEASONALITY + PERIOD–GATED STUB)
# ============================================================

from forecastiq.engine.afe.afe_contract import (
    AFECommittedDataset,
    AFEDatasetIntelligence,
)
from forecastiq.engine.afe.afe_result_schema import ForecastOutput


def run_sarima(
    dataset: AFECommittedDataset,
    intelligence: AFEDatasetIntelligence,
    horizon: int,
    seasonal_period: int,
) -> ForecastOutput:
    """
    SARIMA forecast stub with strict seasonality + explicit period governance.

    GOVERNANCE RULES (NON-NEGOTIABLE):
    - SARIMA REQUIRES confirmed seasonality
    - SARIMA REQUIRES explicit seasonal_period input
    - seasonal_period MUST exist in intelligence.dominant_periods
    - No implicit inference
    - No fallback to ARIMA
    - No auto-configuration

    This function exists ONLY to:
    - Enforce SARIMA eligibility rules
    - Enforce explicit seasonal-period selection
    - Preserve model identity in AFE execution plans
    """

    # --------------------------------------------------------
    # HARD GOVERNANCE GATE 1: SEASONALITY REQUIRED
    # --------------------------------------------------------

    if not intelligence.seasonality_detected:
        raise NotImplementedError(
            "SARIMA execution blocked: no seasonality detected "
            "by Dataset Intelligence. Model not eligible."
        )

    # --------------------------------------------------------
    # HARD GOVERNANCE GATE 2: DOMINANT PERIODS REQUIRED
    # --------------------------------------------------------

    if not intelligence.dominant_periods:
        raise NotImplementedError(
            "SARIMA execution blocked: seasonality detected but no "
            "dominant seasonal periods declared."
        )

    # --------------------------------------------------------
    # HARD GOVERNANCE GATE 3: EXPLICIT PERIOD REQUIRED
    # --------------------------------------------------------

    if seasonal_period is None:
        raise NotImplementedError(
            "SARIMA execution blocked: explicit seasonal_period "
            "must be provided. No implicit selection permitted."
        )

    if seasonal_period not in intelligence.dominant_periods:
        raise NotImplementedError(
            f"SARIMA execution blocked: seasonal_period={seasonal_period} "
            f"is not present in detected dominant periods "
            f"{intelligence.dominant_periods}."
        )

    # --------------------------------------------------------
    # IMPLEMENTATION DEFERRED (INTENTIONAL)
    # --------------------------------------------------------

    raise NotImplementedError(
        "SARIMA execution not yet implemented. "
        "Seasonality and explicit seasonal-period eligibility verified."
    )
