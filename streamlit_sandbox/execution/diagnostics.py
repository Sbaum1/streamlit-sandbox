# FILE: execution/diagnostics.py
# ROLE: FORECAST DIAGNOSTICS ENGINE (EXECUTIVE-GRADE)
# STATUS: CANONICAL / ENGINE-SAFE
#
# GOVERNANCE:
# - Deterministic only
# - No model ranking or scoring
# - No UI logic
# - Diagnostics must NEVER block forecasts
# - Safe with partial or missing CI data
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


# ==================================================
# CORE DIAGNOSTIC HELPERS
# ==================================================

def _safe_autocorr(series: pd.Series, lag: int = 1) -> float | None:
    try:
        if len(series) <= lag:
            return None
        return float(series.autocorr(lag=lag))
    except Exception:
        return None


def _safe_std(series: pd.Series) -> float | None:
    try:
        return float(series.std())
    except Exception:
        return None


def _safe_mean(series: pd.Series) -> float | None:
    try:
        return float(series.mean())
    except Exception:
        return None


# ==================================================
# MAIN DIAGNOSTIC ENTRYPOINT
# ==================================================

def run_diagnostics(
    forecast_df: pd.DataFrame,
    confidence_level: float | None = None,
) -> Dict[str, Any]:
    """
    Runs executive-grade diagnostics on a model forecast output.

    Required columns:
    - actual
    - forecast

    Optional columns:
    - ci_low
    - ci_high
    """

    diagnostics: Dict[str, Any] = {
        "residuals": {},
        "stability": {},
        "confidence": {},
        "regime": {},
        "executive_flags": [],
    }

    # -------------------------------
    # Align historical actuals
    # -------------------------------
    hist = forecast_df.dropna(subset=["actual", "forecast"]).copy()
    if hist.empty or len(hist) < 5:
        diagnostics["executive_flags"].append("INSUFFICIENT_HISTORY")
        return diagnostics

    actual = hist["actual"].astype(float)
    forecast = hist["forecast"].astype(float)
    residuals = forecast - actual

    # -------------------------------
    # Residual diagnostics
    # -------------------------------
    diagnostics["residuals"] = {
        "mean_error": _safe_mean(residuals),
        "residual_volatility": _safe_std(residuals),
        "autocorr_lag1": _safe_autocorr(residuals, lag=1),
    }

    if abs(diagnostics["residuals"]["mean_error"] or 0) > actual.std():
        diagnostics["executive_flags"].append("BIAS_DETECTED")

    if abs(diagnostics["residuals"]["autocorr_lag1"] or 0) > 0.3:
        diagnostics["executive_flags"].append("AUTOCORRELATED_ERRORS")

    # -------------------------------
    # Stability: early vs late split
    # -------------------------------
    split = int(len(residuals) * 0.5)
    early = residuals.iloc[:split]
    late = residuals.iloc[split:]

    early_mae = np.mean(np.abs(early))
    late_mae = np.mean(np.abs(late))

    diagnostics["stability"] = {
        "early_mae": float(early_mae),
        "late_mae": float(late_mae),
        "mae_change_pct": float(
            (late_mae - early_mae) / early_mae
        ) if early_mae > 0 else None,
    }

    if early_mae > 0 and late_mae > early_mae * 1.25:
        diagnostics["executive_flags"].append("DEGRADING_ACCURACY")

    # -------------------------------
    # Confidence calibration
    # -------------------------------
    if {"ci_low", "ci_high"}.issubset(hist.columns):
        try:
            inside = (
                (actual >= hist["ci_low"].astype(float))
                & (actual <= hist["ci_high"].astype(float))
            )
            hit_rate = inside.mean()

            diagnostics["confidence"] = {
                "interval_hit_rate": float(hit_rate),
                "expected_level": confidence_level,
            }

            if confidence_level is not None:
                if hit_rate < confidence_level * 0.8:
                    diagnostics["executive_flags"].append("UNDERCONFIDENT_INTERVALS")
                elif hit_rate > confidence_level * 1.2:
                    diagnostics["executive_flags"].append("OVERCONFIDENT_INTERVALS")

        except Exception:
            diagnostics["confidence"] = {"interval_hit_rate": None}

    else:
        diagnostics["confidence"] = {"interval_hit_rate": None}

    # -------------------------------
    # Regime behavior (volatility shift)
    # -------------------------------
    early_vol = early.std()
    late_vol = late.std()

    diagnostics["regime"] = {
        "early_volatility": float(early_vol),
        "late_volatility": float(late_vol),
        "volatility_ratio": float(late_vol / early_vol) if early_vol > 0 else None,
    }

    if early_vol > 0 and late_vol > early_vol * 1.5:
        diagnostics["executive_flags"].append("POSSIBLE_REGIME_SHIFT")

    # -------------------------------
    # Executive summary signal
    # -------------------------------
    if not diagnostics["executive_flags"]:
        diagnostics["executive_flags"].append("STABLE")

    return diagnostics

