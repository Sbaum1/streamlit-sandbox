# FILE: analysis/diagnostics.py
# ROLE: FORECAST DIAGNOSTICS ENGINE
# STATUS: EXECUTIVE-GRADE / HARDENED
# ==================================================
#
# PURPOSE:
#   Provide conservative, decision-safe diagnostic
#   signals to assess forecast credibility.
#
# CORE PRINCIPLE:
#   Absence of evidence is NOT evidence of stability.
#
# GOVERNANCE:
#   - Deterministic
#   - Read-only
#   - Non-blocking
#   - Defensive by default
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


# ==================================================
# PUBLIC ENTRYPOINT
# ==================================================

def compute_diagnostics(
    forecast_df: pd.DataFrame,
    metrics: Dict[str, float],
) -> Dict[str, Any]:

    diagnostics: Dict[str, Any] = {}

    df = forecast_df.copy()

    # --------------------------------------------------
    # HISTORICAL ROWS ONLY
    # --------------------------------------------------
    hist = df.dropna(subset=["actual", "forecast"])
    n_obs = len(hist)

    diagnostics["n_observations"] = n_obs

    if n_obs < 6:
        diagnostics["notes"] = (
            "Insufficient historical observations for reliable diagnostics."
        )
        diagnostics["bias_direction"] = "Unknown"
        diagnostics["stability_score"] = None
        diagnostics["ci_coverage"] = None
        diagnostics["regime_flag"] = "Insufficient history"
        return diagnostics

    actual = hist["actual"].astype(float)
    forecast = hist["forecast"].astype(float)
    errors = forecast - actual

    abs_actual_mean = max(actual.abs().mean(), 1.0)

    # --------------------------------------------------
    # BIAS DIRECTION
    # --------------------------------------------------
    bias = metrics.get("Bias")

    if bias is None or bias != bias:
        diagnostics["bias_direction"] = "Unknown"
    elif bias > 0:
        diagnostics["bias_direction"] = "Over-forecasting"
    elif bias < 0:
        diagnostics["bias_direction"] = "Under-forecasting"
    else:
        diagnostics["bias_direction"] = "Neutral"

    # --------------------------------------------------
    # STABILITY (NORMALIZED ERROR VOLATILITY)
    # --------------------------------------------------
    if len(errors) > 1:
        raw_std = float(np.std(errors))
        diagnostics["stability_score"] = raw_std / abs_actual_mean
    else:
        diagnostics["stability_score"] = None

    # --------------------------------------------------
    # CONFIDENCE INTERVAL CALIBRATION (DEFENSIVE)
    # --------------------------------------------------
    if {"ci_low", "ci_high"}.issubset(hist.columns):
        ci_hist = hist.dropna(subset=["ci_low", "ci_high"])
        ci_n = len(ci_hist)

        if ci_n < max(10, n_obs * 0.5):
            diagnostics["ci_coverage"] = None
            diagnostics["ci_note"] = "Insufficient data to validate confidence intervals."
        else:
            low = ci_hist["ci_low"].astype(float)
            high = ci_hist["ci_high"].astype(float)
            within = (ci_hist["actual"] >= low) & (ci_hist["actual"] <= high)
            diagnostics["ci_coverage"] = float(within.mean())
    else:
        diagnostics["ci_coverage"] = None

    # --------------------------------------------------
    # REGIME FLAGGING (CONSERVATIVE)
    # --------------------------------------------------
    diagnostics["regime_flag"] = _detect_regime_shift(errors)

    # --------------------------------------------------
    # EXECUTIVE NOTES
    # --------------------------------------------------
    diagnostics["notes"] = _build_notes(diagnostics)

    return diagnostics


# ==================================================
# REGIME DETECTION (CONSERVATIVE)
# ==================================================

def _detect_regime_shift(errors: pd.Series) -> str:
    if len(errors) < 12:
        return "Insufficient history"

    rolling_mean = errors.rolling(window=6).mean()

    early = rolling_mean.iloc[: len(rolling_mean) // 2].dropna()
    late = rolling_mean.iloc[len(rolling_mean) // 2 :].dropna()

    if early.empty or late.empty:
        return "No signal"

    delta = late.mean() - early.mean()
    threshold = np.std(errors)

    if threshold > 0 and abs(delta) > threshold:
        return "Possible structural change"

    return "No regime shift detected"


# ==================================================
# EXECUTIVE NOTE BUILDER
# ==================================================

def _build_notes(diagnostics: Dict[str, Any]) -> str:
    notes = []

    if diagnostics.get("n_observations", 0) < 12:
        notes.append("Limited historical data reduces diagnostic confidence.")

    bias = diagnostics.get("bias_direction")
    if bias in {"Over-forecasting", "Under-forecasting"}:
        notes.append(f"Systematic {bias.lower()} observed.")

    coverage = diagnostics.get("ci_coverage")
    if coverage is None:
        notes.append("Confidence interval calibration could not be validated.")
    else:
        if coverage < 0.6:
            notes.append("Confidence intervals appear too narrow.")
        elif coverage > 0.95:
            notes.append("Confidence intervals appear conservative.")

    regime = diagnostics.get("regime_flag")
    if regime == "Possible structural change":
        notes.append("Error behavior shifted versus earlier history.")

    return " ".join(notes) if notes else "No material diagnostic concerns detected."

