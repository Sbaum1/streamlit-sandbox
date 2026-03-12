# FILE: execution/metrics.py
# ROLE: FORECAST ERROR METRICS + EXECUTIVE DIAGNOSTICS (DETERMINISTIC)
# STATUS: LOCKED / ENGINE-SAFE
#
# GOVERNANCE:
# - Metrics must never crash a model
# - Lengths must be aligned explicitly
# - NaNs are dropped symmetrically
# - Metric names are canonical and stable
# - Diagnostics are derived ONLY (no refits, no ranking)
# ==================================================

import numpy as np
import pandas as pd


# ==================================================
# CORE ALIGNMENT UTILITIES
# ==================================================

def _align_series(
    actual: pd.Series,
    forecast: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Aligns actual and forecast safely.
    Drops NaNs symmetrically and enforces equal length.
    """

    a = pd.Series(actual).astype(float)
    f = pd.Series(forecast).astype(float)

    df = pd.DataFrame({"actual": a, "forecast": f}).dropna()

    if df.empty:
        raise ValueError("No overlapping non-null values to compute metrics.")

    return df["actual"], df["forecast"]


def _safe_pct(error: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Safe percentage error (no division by zero explosions).
    """
    return np.where(actual == 0, 0.0, error / actual)


# ==================================================
# CORE METRICS (CANONICAL)
# ==================================================

def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """
    Computes deterministic forecast accuracy metrics.

    Canonical metric keys:
    - MAE
    - RMSE
    - MAPE
    - Bias
    """

    actual_aligned, forecast_aligned = _align_series(actual, forecast)

    err = forecast_aligned - actual_aligned

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(
        np.mean(np.abs(_safe_pct(err.values, actual_aligned.values))) * 100.0
    )
    bias = float(np.mean(err))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias,
    }


# ==================================================
# EXECUTIVE DECISION DIAGNOSTICS
# ==================================================

def compute_diagnostics(actual: pd.Series, forecast: pd.Series) -> dict:
    """
    Computes executive-grade diagnostic signals.

    These are NOT ranking metrics.
    They answer: "When should I trust this model?"

    Diagnostic keys:
    - Bias_Direction
    - Bias_Magnitude
    - Stability_Score
    - Volatility_Sensitivity
    - Shock_Exposure
    """

    actual_aligned, forecast_aligned = _align_series(actual, forecast)
    err = forecast_aligned - actual_aligned

    # -------------------------------
    # Bias diagnostics
    # -------------------------------
    bias = float(np.mean(err))
    bias_direction = (
        "Over-Forecasting" if bias > 0
        else "Under-Forecasting" if bias < 0
        else "Neutral"
    )

    # -------------------------------
    # Stability score
    # (Lower = more stable residual behavior)
    # -------------------------------
    stability_score = float(np.std(err))

    # -------------------------------
    # Volatility sensitivity
    # Correlation between absolute error and absolute actual change
    # -------------------------------
    actual_diff = actual_aligned.diff().abs().iloc[1:]
    error_abs = err.abs().iloc[1:]

    if len(actual_diff) > 3:
        volatility_sensitivity = float(
            np.corrcoef(actual_diff, error_abs)[0, 1]
        )
    else:
        volatility_sensitivity = np.nan

    # -------------------------------
    # Shock exposure
    # Share of total error driven by top 10% largest misses
    # -------------------------------
    abs_err = err.abs()
    if len(abs_err) > 0:
        threshold = np.quantile(abs_err, 0.9)
        shock_exposure = float(
            abs_err[abs_err >= threshold].sum() / abs_err.sum()
        )
    else:
        shock_exposure = np.nan

    return {
        "Bias_Direction": bias_direction,
        "Bias_Magnitude": float(abs(bias)),
        "Stability_Score": stability_score,
        "Volatility_Sensitivity": volatility_sensitivity,
        "Shock_Exposure": shock_exposure,
    }


# ==================================================
# COMBINED METRICS + DIAGNOSTICS
# ==================================================

def compute_metrics_with_diagnostics(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict:
    """
    Convenience wrapper used by the UI layer.

    Returns:
    - All canonical metrics
    - All executive diagnostics
    """

    metrics = compute_metrics(actual, forecast)
    diagnostics = compute_diagnostics(actual, forecast)

    return {
        **metrics,
        **diagnostics,
    }

