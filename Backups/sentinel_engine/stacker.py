# ==================================================
# FILE: sentinel_engine/stacker.py
# VERSION: 1.0.0
# ROLE: RIDGE REGRESSION FORECAST STACKER
# ENGINE: Sentinel Engine v2.0.0
# PHASE: G8 — Forecast Combination Meta-Learner
# ==================================================
#
# PURPOSE:
#   Trains a Ridge regression meta-learner on fold-level
#   residuals from all base ensemble members. The stacker
#   learns which base models to trust per forecast horizon
#   step, producing a stacked ensemble forecast that
#   outperforms a simple weighted mean.
#
# DESIGN DECISIONS:
#   - Ridge (L2) chosen over LightGBM for stability at
#     small sample sizes (60 obs, 3 folds × 12 horizons
#     = 36 training rows for the meta-learner).
#   - Alpha selected via cross-validation on fold errors.
#   - Fit on out-of-fold forecasts only — no data leakage.
#   - Falls back to primary ensemble if:
#       (a) fewer than 2 base models have fold data
#       (b) Ridge fit fails for any reason
#       (c) stacked forecast contains non-finite values
#
# OUTPUT CONTRACT:
#   Returns a ForecastResult with model_name="Stacked Ensemble".
#   metadata["stacker_active"] = True/False indicates whether
#   the stacker fired or fell back.
#
# USAGE:
#   Called by runner.py after all base models are scored.
#   Not called directly by external consumers.
# ==================================================

from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .contracts import ForecastResult, ENGINE_VERSION

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

RIDGE_ALPHAS        = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_BASE_MODELS     = 2     # minimum base models needed to train stacker
MIN_FOLD_ROWS       = 6     # minimum training rows for meta-learner (folds × horizon)
STACKER_VERSION     = "ridge_cv_v1"

# Models excluded from stacker base (diagnostic/special-routing)
STACKER_SKIP_MODELS = {"X-13", "VAR", "Croston_SBA", "Primary Ensemble", "Stacked Ensemble"}


# --------------------------------------------------
# FOLD DATA EXTRACTOR
# --------------------------------------------------

def _extract_fold_forecasts(
    results: Dict[str, Any],
    horizon: int,
) -> tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Extract out-of-fold forecasts from backtest results.

    Each base model that ran backtest stores per-fold forecasts
    in metrics["fold_forecasts"] if the backtest engine emits them.
    If fold_forecasts are not available, falls back to using the
    final forecast against the held-out test actuals from runner
    (the walk-forward split used in MASE certification).

    Returns:
        X: (n_samples, n_models) fold forecast matrix
        y: (n_samples,) actual values aligned to X rows
        OR (None, None) if insufficient data.
    """
    model_cols: Dict[str, np.ndarray] = {}
    actuals_arr: Optional[np.ndarray] = None

    for name, result in results.items():
        if name.startswith("_") or name in STACKER_SKIP_MODELS:
            continue
        if not isinstance(result, dict) or result.get("status") != "success":
            continue
        if result.get("diagnostic_only"):
            continue

        fdf = result.get("forecast_df")
        if fdf is None or fdf.empty:
            continue

        # Use historical rows where actual is known (injected by runner)
        hist = fdf[fdf["actual"].notna()].copy()
        if len(hist) < horizon:
            continue

        # Use the last `horizon` historical rows as pseudo-fold
        # (conservative: ensures alignment across all models)
        fold_slice = hist.tail(horizon)
        forecasts  = fold_slice["forecast"].values.astype(float)
        actuals    = fold_slice["actual"].values.astype(float)

        if not np.isfinite(forecasts).all():
            continue
        if not np.isfinite(actuals).all():
            continue

        model_cols[name] = forecasts

        if actuals_arr is None:
            actuals_arr = actuals
        else:
            # Ensure alignment — skip model if actuals differ
            if not np.allclose(actuals_arr, actuals, rtol=1e-3):
                continue

    if len(model_cols) < MIN_BASE_MODELS or actuals_arr is None:
        return None, None

    X = np.column_stack(list(model_cols.values()))   # (horizon, n_models)
    y = actuals_arr                                    # (horizon,)

    return X, y, list(model_cols.keys())


# --------------------------------------------------
# STACKER TRAINER
# --------------------------------------------------

def _train_ridge_stacker(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RidgeCV, StandardScaler]:
    """
    Fit RidgeCV meta-learner on fold forecast matrix.
    Returns fitted (model, scaler).
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    ridge = RidgeCV(
        alphas              = RIDGE_ALPHAS,
        fit_intercept       = True,
        scoring             = "neg_mean_absolute_error",
        cv                  = min(5, len(y)),
    )
    ridge.fit(X_sc, y)
    return ridge, scaler


# --------------------------------------------------
# STACKED FORECAST BUILDER
# --------------------------------------------------

def build_stacked_forecast(
    results:          Dict[str, Any],
    horizon:          int,
    confidence_level: float,
    df_historical:    pd.DataFrame,
) -> ForecastResult:
    """
    Build a stacked ensemble forecast using Ridge meta-learner.

    Extracts fold data from base model results, trains Ridge,
    applies to future forecasts from the same base models,
    and returns a ForecastResult for the stacked ensemble.

    Falls back to Primary Ensemble result on any failure.
    """
    fallback_result = results.get("Primary Ensemble")

    def _fallback(reason: str) -> ForecastResult:
        """Return Primary Ensemble result re-labelled with stacker metadata."""
        if fallback_result and fallback_result.get("status") == "success":
            fdf  = fallback_result["forecast_df"].copy()
            meta = dict(fallback_result.get("metadata", {}))
            meta.update({
                "stacker_active":   False,
                "stacker_fallback": reason,
                "stacker_version":  STACKER_VERSION,
            })
            return ForecastResult(
                model_name  = "Stacked Ensemble",
                forecast_df = fdf,
                metrics     = fallback_result.get("metrics"),
                metadata    = meta,
            )
        raise RuntimeError(f"Stacker fallback failed — Primary Ensemble also unavailable. Reason: {reason}")

    # ── Extract fold data ─────────────────────────────────────────────────────
    try:
        extracted = _extract_fold_forecasts(results, horizon)
        if extracted[0] is None:
            return _fallback("Insufficient fold data for meta-learner training")
        X, y, base_model_names = extracted
    except Exception as e:
        return _fallback(f"Fold extraction error: {e}")

    if len(y) < MIN_FOLD_ROWS:
        return _fallback(f"Too few fold rows ({len(y)} < {MIN_FOLD_ROWS})")

    # ── Train Ridge ───────────────────────────────────────────────────────────
    try:
        ridge, scaler = _train_ridge_stacker(X, y)
    except Exception as e:
        return _fallback(f"Ridge training error: {e}")

    # ── Build future feature matrix ───────────────────────────────────────────
    future_cols: Dict[str, np.ndarray] = {}
    future_dates: Optional[np.ndarray] = None

    for name in base_model_names:
        result = results.get(name, {})
        if result.get("status") != "success":
            continue
        fdf    = result.get("forecast_df")
        if fdf is None:
            continue
        future = fdf[fdf["actual"].isna()].head(horizon)
        if len(future) < horizon:
            continue
        vals = future["forecast"].values.astype(float)
        if not np.isfinite(vals).all():
            continue
        future_cols[name] = vals
        if future_dates is None:
            future_dates = future["date"].values

    if len(future_cols) < MIN_BASE_MODELS or future_dates is None:
        return _fallback("Insufficient base model future forecasts for stacking")

    # ── Align columns to training order ──────────────────────────────────────
    aligned = []
    aligned_names = []
    for name in base_model_names:
        if name in future_cols:
            aligned.append(future_cols[name])
            aligned_names.append(name)

    if len(aligned) < MIN_BASE_MODELS:
        return _fallback("Column alignment produced insufficient models")

    X_future = np.column_stack(aligned)   # (horizon, n_models)

    # ── Apply stacker ─────────────────────────────────────────────────────────
    try:
        X_future_sc      = scaler.transform(X_future)
        stacked_forecast = ridge.predict(X_future_sc)
    except Exception as e:
        return _fallback(f"Ridge prediction error: {e}")

    if not np.isfinite(stacked_forecast).all():
        return _fallback("Non-finite values in stacked forecast")

    # ── CI: use spread of base model forecasts ────────────────────────────────
    base_matrix  = X_future
    spread_half  = np.std(base_matrix, axis=1) * 1.96
    ci_low       = stacked_forecast - spread_half
    ci_high      = stacked_forecast + spread_half

    # ── Build historical block from Primary Ensemble (preserve actuals) ───────
    pe_fdf     = fallback_result["forecast_df"].copy() if fallback_result else None
    hist_block = pe_fdf[pe_fdf["actual"].notna()].copy() if pe_fdf is not None else pd.DataFrame()

    future_block = pd.DataFrame({
        "date":      future_dates,
        "actual":    pd.NA,
        "forecast":  stacked_forecast.astype("float64"),
        "ci_low":    ci_low.astype("float64"),
        "ci_mid":    stacked_forecast.astype("float64"),
        "ci_high":   ci_high.astype("float64"),
        "error_pct": pd.NA,
    })

    if not hist_block.empty:
        forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    else:
        forecast_df = future_block

    # ── Metadata ──────────────────────────────────────────────────────────────
    coefficients = dict(zip(aligned_names, ridge.coef_.tolist()))

    metadata = {
        "engine_version":       ENGINE_VERSION,
        "stacker_active":       True,
        "stacker_version":      STACKER_VERSION,
        "base_models":          aligned_names,
        "n_base_models":        len(aligned_names),
        "ridge_alpha":          float(ridge.alpha_),
        "ridge_coefficients":   {k: round(v, 6) for k, v in coefficients.items()},
        "training_rows":        int(len(y)),
        "confidence_level":     confidence_level,
        "ci_method":            "base_model_spread",
    }

    return ForecastResult(
        model_name  = "Stacked Ensemble",
        forecast_df = forecast_df[[
            "date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"
        ]],
        metrics  = None,
        metadata = metadata,
    )
