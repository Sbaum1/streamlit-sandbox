# ============================================================
# FILE: change_point_detection.py
# ROLE: CHANGE POINT / STRUCTURAL BREAK DETECTION
# STATUS: AFE STRUCTURAL MODEL — GOVERNED / PRODUCTION-GRADE
# ============================================================

from typing import List, Dict
import numpy as np

from forecastiq.engine.afe.afe_contract import AFECommittedDataset
from forecastiq.engine.afe.afe_result_schema import StructuralOutput


def run_change_point_detection(
    dataset: AFECommittedDataset,
    window: int = 12,
    mean_shift_threshold: float = 1.5,
    variance_shift_threshold: float = 1.5,
) -> StructuralOutput:
    """
    Deterministic Change Point / Structural Break Detection.

    GOVERNANCE (LOCKED):
    - Diagnostic-only (no forecasts)
    - Fixed rolling window
    - Fixed thresholds (no optimization)
    - Deterministic, auditable
    """

    if not isinstance(dataset, AFECommittedDataset):
        raise TypeError(
            "Change Point Detection requires an AFECommittedDataset instance."
        )

    values: List[float] = dataset.values

    if len(values) < window * 2:
        raise ValueError(
            "Insufficient data for change point detection."
        )

    series = np.array(values, dtype=float)

    # --------------------------------------------------------
    # ROLLING WINDOW STATISTICS
    # --------------------------------------------------------

    recent = series[-window:]
    prior = series[-2 * window : -window]

    recent_mean = np.mean(recent)
    prior_mean = np.mean(prior)

    recent_var = np.var(recent)
    prior_var = np.var(prior)

    # --------------------------------------------------------
    # STRUCTURAL SHIFT DETECTION
    # --------------------------------------------------------

    mean_shift_ratio = (
        abs(recent_mean - prior_mean) / (np.std(prior) + 1e-8)
    )
    variance_shift_ratio = (
        recent_var / (prior_var + 1e-8)
    )

    mean_shift = mean_shift_ratio >= mean_shift_threshold
    variance_shift = variance_shift_ratio >= variance_shift_threshold

    structural_break_detected = mean_shift or variance_shift

    # Regime shift probability (bounded, interpretable heuristic)
    regime_shift_probability = float(
        min(
            1.0,
            0.5 * (mean_shift_ratio / mean_shift_threshold)
            + 0.5 * (variance_shift_ratio / variance_shift_threshold),
        )
    )

    signals: Dict[str, float] = {
        "structural_break_detected": float(structural_break_detected),
        "mean_shift_ratio": float(mean_shift_ratio),
        "variance_shift_ratio": float(variance_shift_ratio),
        "regime_shift_probability": regime_shift_probability,
    }

    narrative = (
        "Deterministic change point analysis completed. "
        f"Mean shift ratio={mean_shift_ratio:.2f}, "
        f"Variance shift ratio={variance_shift_ratio:.2f}. "
        "Signals reflect rolling-window structural comparison only."
    )

    return StructuralOutput(
        signals=signals,
        narrative=narrative,
    )
