# ==================================================
# FILE: sentinel_engine/models/prophet.py
# VERSION: 3.0.0
# MODEL: PROPHET (Multiplicative, Regime-Change Tuned)
# ENGINE: Sentinel Engine v2.0.0
# STATUS: FORTUNE 100 REBUILD — REGIME-CHANGE TUNED — 3B-3 v2
# UPDATED: G1 — Regime-detection pre-filter
# ==================================================
#
# 3B-3 v2 TUNING — REGIME CHANGE FIX (REVISED):
#
#   v1 failure: Explicit changepoint anchors (2020-03, 2022-01)
#               caused Prophet to overfit the 2022 inflection
#               and extrapolate the reversal too aggressively.
#               MASE worsened from 1.3072 → 2.0410.
#
#   v2 approach: Remove explicit anchors entirely.
#               Rely on tuned hyperparameters to improve
#               regime sensitivity without overfitting.
#
#   Fix 1: changepoint_prior_scale 0.05 → 0.15
#           More responsive to trend inflections than default.
#           Less aggressive than v1 (0.20) which overcorrected.
#
#   Fix 2: changepoint_range 0.80 → 0.90
#           Extends detection window to 90% of training data.
#           Captures late-series inflections without anchoring.
#
#   Fix 3: seasonality_mode additive → multiplicative
#           Proportional seasonal variation handled correctly.
#           Standard for financial/economic monthly series.
#
#   Fix 4: seasonality_prior_scale 10.0 → 5.0
#           Moderate seasonal flexibility. Prevents overfitting
#           seasonal component at expense of trend accuracy.
#
#   Tuning version: regime_change_v2
#   Auditable via metadata["tuning_version"]
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

from sentinel_engine.models.contracts import ForecastResult


# --------------------------------------------------
# REGIME CHANGE TUNING CONSTANTS — v2
# --------------------------------------------------

CHANGEPOINT_PRIOR_SCALE  = 0.15
CHANGEPOINT_RANGE        = 0.90
SEASONALITY_MODE         = "multiplicative"
SEASONALITY_PRIOR_SCALE  = 5.0
UNCERTAINTY_SAMPLES      = 1000

TUNING_VERSION = "regime_change_v2"

# --------------------------------------------------
# REGIME DETECTION CONSTANTS — G1
# --------------------------------------------------
# A structural break is flagged when a rolling window
# shows a mean shift exceeding BREAK_SIGMA_THRESHOLD
# standard deviations relative to the full-series std.
# BREAK_LOOKBACK controls how many months are examined
# at the tail of training data.
#
# On COVID data: the 2020 window has a mean shift of
# ~4–6 sigma — well above the 2.5 threshold.
# On stable series: shifts are typically < 1 sigma.
# --------------------------------------------------

BREAK_LOOKBACK       = 24   # months to examine at tail
BREAK_WINDOW         = 6    # rolling window size for mean
BREAK_SIGMA_THRESHOLD = 2.5  # sigma threshold for suppression


# --------------------------------------------------
# REGIME DETECTOR
# --------------------------------------------------

def _detect_structural_break(values: np.ndarray) -> tuple[bool, float]:
    """
    Detect a structural break in the tail of a series.

    Method: rolling-window mean shift vs full-series std.
    Returns (break_detected: bool, max_shift_sigma: float).
    """
    if len(values) < BREAK_LOOKBACK + BREAK_WINDOW:
        return False, 0.0

    full_std = float(np.std(values))
    if full_std < 1e-8:
        return False, 0.0

    tail   = values[-BREAK_LOOKBACK:]
    shifts = []
    for i in range(len(tail) - BREAK_WINDOW + 1):
        window_mean = float(np.mean(tail[i:i + BREAK_WINDOW]))
        pre_mean    = float(np.mean(values[:-BREAK_LOOKBACK + i] if i > 0 else values[:-BREAK_LOOKBACK]))
        shift_sigma = abs(window_mean - pre_mean) / full_std
        shifts.append(shift_sigma)

    max_shift = float(max(shifts)) if shifts else 0.0
    return max_shift >= BREAK_SIGMA_THRESHOLD, max_shift


# ==================================================
# MODEL RUNNER
# ==================================================

def run_prophet(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Prophet requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date")

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"Prophet requires monthly frequency. Detected: {inferred}")

    if df["value"].isna().any():
        raise ValueError("Missing values detected in input series.")

    y = df["value"].astype("float64")

    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")

    if len(df) < 24:
        raise ValueError("Minimum 24 observations required.")

    # --------------------------------------------------
    # G1 — REGIME DETECTION PRE-FILTER
    # --------------------------------------------------
    # If a structural break >= BREAK_SIGMA_THRESHOLD sigma
    # is detected in the last BREAK_LOOKBACK months,
    # Prophet is suppressed and returns a graceful skip.
    # Prevents Prophet from extrapolating regime shocks.
    # Suppression is logged in metadata — fully auditable.
    # --------------------------------------------------

    break_detected, max_shift_sigma = _detect_structural_break(y.values)
    if break_detected:
        last_val   = float(y.iloc[-1])
        last_date  = df["date"].iloc[-1]
        hist_dates = df["date"].values
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=horizon, freq="MS",
        )
        hist_block = pd.DataFrame({
            "date": hist_dates, "actual": np.nan,
            "forecast": df["value"].values.astype("float64"),
            "ci_low":   df["value"].values.astype("float64"),
            "ci_mid":   df["value"].values.astype("float64"),
            "ci_high":  df["value"].values.astype("float64"),
            "error_pct": np.nan,
        })
        future_block = pd.DataFrame({
            "date": future_dates, "actual": np.nan,
            "forecast": last_val, "ci_low": last_val,
            "ci_mid": last_val, "ci_high": last_val,
            "error_pct": np.nan,
        })
        forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
        return ForecastResult(
            model_name  = "Prophet",
            forecast_df = forecast_df[
                ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
            ],
            metrics  = None,
            metadata = {
                "regime_suppressed":      True,
                "max_shift_sigma":        round(max_shift_sigma, 3),
                "break_lookback":         BREAK_LOOKBACK,
                "break_sigma_threshold":  BREAK_SIGMA_THRESHOLD,
                "suppression_reason":     (
                    f"Structural break detected ({max_shift_sigma:.2f}σ >= "
                    f"{BREAK_SIGMA_THRESHOLD}σ). Prophet suppressed."
                ),
                "tuning_version":         TUNING_VERSION,
                "compliance":             "Fortune_100_standard",
            },
        )

    # --------------------------------------------------
    # PROPHET MODEL — REGIME-CHANGE TUNED v2
    # --------------------------------------------------

    prophet_df = df.rename(columns={"date": "ds", "value": "y"})[["ds", "y"]]

    model = Prophet(
        interval_width          = confidence_level,
        uncertainty_samples     = UNCERTAINTY_SAMPLES,
        changepoint_prior_scale = CHANGEPOINT_PRIOR_SCALE,
        changepoint_range       = CHANGEPOINT_RANGE,
        seasonality_mode        = SEASONALITY_MODE,
        seasonality_prior_scale = SEASONALITY_PRIOR_SCALE,
        daily_seasonality       = False,
        weekly_seasonality      = False,
        yearly_seasonality      = True,
    )

    model.fit(prophet_df)

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    fitted_hist = model.predict(prophet_df)

    if fitted_hist[["yhat", "yhat_lower", "yhat_upper"]].isna().any().any():
        raise RuntimeError("Invalid fitted output detected.")

    hist_block = pd.DataFrame(
        {
            "date":      fitted_hist["ds"],
            "actual":    np.nan,
            "forecast":  fitted_hist["yhat"].astype("float64").values,
            "ci_low":    fitted_hist["yhat_lower"].astype("float64").values,
            "ci_mid":    fitted_hist["yhat"].astype("float64").values,
            "ci_high":   fitted_hist["yhat_upper"].astype("float64").values,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    future = model.make_future_dataframe(
        periods         = horizon,
        freq            = inferred,
        include_history = False,
    )

    forecast_future = model.predict(future)

    if forecast_future[["yhat", "yhat_lower", "yhat_upper"]].isna().any().any():
        raise RuntimeError("Invalid future forecast output detected.")

    if not np.isfinite(forecast_future["yhat"]).all():
        raise RuntimeError("Non-finite forecast values detected.")

    if forecast_future["ds"].min() <= df["date"].max():
        raise RuntimeError("Forecast horizon overlaps historical data.")

    future_block = pd.DataFrame(
        {
            "date":      forecast_future["ds"],
            "actual":    np.nan,
            "forecast":  forecast_future["yhat"].astype("float64").values,
            "ci_low":    forecast_future["yhat_lower"].astype("float64").values,
            "ci_mid":    forecast_future["yhat"].astype("float64").values,
            "ci_high":   forecast_future["yhat_upper"].astype("float64").values,
            "error_pct": np.nan,
        }
    )

    # --------------------------------------------------
    # DTYPE GOVERNANCE
    # --------------------------------------------------

    numeric_cols = ["forecast", "ci_low", "ci_mid", "ci_high"]
    hist_block[numeric_cols]   = hist_block[numeric_cols].astype("float64")
    future_block[numeric_cols] = future_block[numeric_cols].astype("float64")

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)

    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")

    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "Prophet",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "trend":                    "piecewise_linear",
            "seasonality":              "yearly",
            "seasonality_mode":         SEASONALITY_MODE,
            "changepoint_prior_scale":  CHANGEPOINT_PRIOR_SCALE,
            "changepoint_range":        CHANGEPOINT_RANGE,
            "seasonality_prior_scale":  SEASONALITY_PRIOR_SCALE,
            "bayesian_ci":              True,
            "uncertainty_samples":      UNCERTAINTY_SAMPLES,
            "frequency":                inferred,
            "confidence_level":         confidence_level,
            "tuning_version":           TUNING_VERSION,
            "compliance":               "Fortune_100_standard",
        },
    )
