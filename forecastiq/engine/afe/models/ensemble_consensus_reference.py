# ============================================================
# FILE: ensemble_consensus_reference.py
# ROLE: ENSEMBLE / CONSENSUS REFERENCE MODEL (ECR)
# STATUS: AFE MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List, Dict, Any
import numpy as np

from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    ForecastInterval,
)


def run_ensemble_consensus_reference(
    model_outputs: List[ForecastOutput],
) -> Dict[str, Any]:
    """
    Ensemble / Consensus Reference — GOVERNED, DIAGNOSTIC-ONLY MODEL.

    GOVERNANCE (LOCKED):
    - This is NOT a forecasting model
    - It does NOT generate independent forecasts
    - It operates ONLY on already-executed model outputs
    - No weighting, ranking, or optimization
    - Deterministic, auditable, and transparent

    PURPOSE:
    - Measure agreement and divergence across models
    - Surface consensus strength and dispersion
    - Provide reference signals for executive interpretation
    """

    # --------------------------------------------------------
    # INPUT VALIDATION (STRICT)
    # --------------------------------------------------------

    if not model_outputs:
        raise ValueError(
            "Ensemble Consensus Reference requires at least one ForecastOutput."
        )

    for m in model_outputs:
        if not isinstance(m, ForecastOutput):
            raise TypeError(
                "All inputs to Ensemble Consensus Reference must be ForecastOutput instances."
            )

        if not m.point_forecast:
            raise ValueError(
                "ForecastOutput instances must contain point_forecast values."
            )

    horizons = {m.horizon for m in model_outputs}
    if len(horizons) != 1:
        raise ValueError(
            "All ForecastOutput inputs must share the same forecast horizon."
        )

    horizon = horizons.pop()

    # --------------------------------------------------------
    # STACK FORECASTS
    # --------------------------------------------------------

    forecasts = np.vstack(
        [np.asarray(m.point_forecast, dtype=float) for m in model_outputs]
    )

    # --------------------------------------------------------
    # CONSENSUS METRICS (DETERMINISTIC)
    # --------------------------------------------------------

    mean_forecast = np.mean(forecasts, axis=0)
    median_forecast = np.median(forecasts, axis=0)
    std_dev = np.std(forecasts, axis=0)

    # Agreement score: bounded, monotonic, interpretable
    agreement_score = float(
        np.mean(1.0 / (1.0 + std_dev))
    )

    diagnostics: Dict[str, Any] = {
        "model": "Ensemble / Consensus Reference",
        "num_models": len(model_outputs),
        "horizon": horizon,
        "agreement_score": agreement_score,
        "mean_std_dev": float(np.mean(std_dev)),
        "interpretation": (
            "Higher agreement_score indicates stronger cross-model consensus. "
            "Lower values indicate divergence across forecasts."
        ),
        "governance_note": (
            "This output is diagnostic-only and must not be used as an "
            "independent forecast within AFE."
        ),
    }

    # --------------------------------------------------------
    # CONSENSUS REFERENCE OUTPUT (NON-PRESCRIPTIVE)
    # --------------------------------------------------------

    intervals = ForecastInterval(
        base=mean_forecast.tolist(),
        upside=(mean_forecast + std_dev).tolist(),
        downside=(mean_forecast - std_dev).tolist(),
    )

    consensus_output = ForecastOutput(
        horizon=horizon,
        point_forecast=median_forecast.tolist(),
        intervals=intervals,
    )

    return {
        "consensus_reference": consensus_output,
        "diagnostics": diagnostics,
    }
