# ==================================================
# FILE: sentinel_engine/models/bsts.py
# VERSION: 3.0.0
# MODEL: BSTS — OPTIMISED VARIANCE PARAMETER SEARCH
# ENGINE: Sentinel Engine v2.1.0
# UPDATED: M1 — Hyperparameter search over state variances
# ==================================================
#
# M1 UPGRADE — OPTIMISED VARIANCE SEARCH:
#
#   Previous (v2.0.0):
#     Single UnobservedComponents fit with statsmodels defaults.
#     The state variances (level, trend, seasonal) are initialised
#     at default values and optimised by a single MLE pass. For
#     many real-world series the default initialisation lands in a
#     poor local optimum — particularly for series with structural
#     breaks or heavy seasonality.
#
#   Fixed (v3.0.0):
#     Multi-start grid search over initial variance scale factors:
#       level_var_scale  ∈ {0.01, 0.1, 1.0}   — how much level can drift
#       trend_var_scale  ∈ {0.001, 0.01, 0.1}  — how much trend can drift
#     Each (level, trend) combination is fitted independently.
#     Best model selected by log-likelihood (higher = better).
#     Total fits: 9. Each takes ~1-2s on monthly series — acceptable.
#
#   Why this matters:
#     BSTS is highly sensitive to variance initialisation.
#     A level_var_scale that is too low locks the level and forces
#     the model to absorb shocks as noise. Too high and the level
#     chases every data point (over-smooth). The grid finds the
#     right balance per series automatically.
#
#   Structural robustness:
#     Uses local linear trend (level + slope) — this is already
#     the right structure for monthly business series. The seasonal
#     component handles M3-style annual seasonality. No change to
#     model structure — only initialisation search is new.
#
# GOVERNANCE:
#   - best_llf and selected variances logged in metadata
#   - Output contract: ForecastResult unchanged
# ==================================================

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from sentinel_engine.models.contracts import ForecastResult

# --------------------------------------------------
# VARIANCE SEARCH GRID
# --------------------------------------------------

LEVEL_VAR_SCALES = [0.01, 0.1, 1.0]
TREND_VAR_SCALES = [0.001, 0.01, 0.1]


def _fit_with_scales(
    y: np.ndarray,
    level_scale: float,
    trend_scale: float,
    inferred: str,
) -> tuple[object | None, float]:
    """
    Fit BSTS with given variance initialisations.
    Returns (result, log_likelihood). Returns (None, -inf) on failure.
    """
    try:
        y_mean = float(np.mean(np.abs(y))) + 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(
                y,
                level="local linear trend",
                seasonal=12,
            )
            # Build start_params near the provided scale factors
            # statsmodels UC uses log-transformed variances internally
            # We can't directly set them but can perturb the optimiser
            # via different starting points using the model's param names
            start = model.start_params.copy()

            # Identify variance parameters (sigma2_* or irregular/level/trend/seasonal)
            param_names = model.param_names
            for i, name in enumerate(param_names):
                if "level" in name and "var" not in name.lower():
                    continue
                if "sigma2_level" in name or name == "sigma2.level":
                    start[i] = y_mean * level_scale
                elif "sigma2_trend" in name or name == "sigma2.trend":
                    start[i] = y_mean * trend_scale
                elif "sigma2_season" in name or "seasonal" in name:
                    start[i] = y_mean * 0.01  # fixed seasonal variance
                elif "sigma2_irregular" in name or "irregular" in name:
                    start[i] = y_mean * 0.1

            start = np.clip(start, 1e-6, None)
            res   = model.fit(start_params=start, disp=False, maxiter=200)

        if not np.isfinite(res.llf):
            return None, float("-inf")
        if res.fittedvalues.isna().any():
            return None, float("-inf")

        return res, float(res.llf)

    except Exception:
        return None, float("-inf")


def _select_best_bsts(
    y: np.ndarray,
    inferred: str,
) -> tuple[object, float, float, float]:
    """
    Grid search over variance scales, return best by log-likelihood.
    Returns (best_result, best_llf, best_level_scale, best_trend_scale).
    """
    best_res, best_llf       = None, float("-inf")
    best_ls,  best_ts        = LEVEL_VAR_SCALES[1], TREND_VAR_SCALES[1]

    for ls in LEVEL_VAR_SCALES:
        for ts in TREND_VAR_SCALES:
            res, llf = _fit_with_scales(y, ls, ts, inferred)
            if res is not None and llf > best_llf:
                best_res, best_llf = res, llf
                best_ls,  best_ts  = ls, ts

    # Final fallback — default statsmodels fit
    if best_res is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model    = UnobservedComponents(y, level="local linear trend", seasonal=12)
                best_res = model.fit(disp=False)
                best_llf = float(best_res.llf)
        except Exception as e:
            raise RuntimeError(f"BSTS fit failed on all initialisations: {e}") from e

    return best_res, best_llf, best_ls, best_ts


# ==================================================
# MODEL RUNNER
# ==================================================

def run_bsts(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("BSTS requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected. Index integrity violated.")

    df = df.sort_values("date").set_index("date")
    inferred = pd.infer_freq(df.index)
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")
    if inferred not in ("MS", "M"):
        raise ValueError(f"BSTS requires monthly frequency. Detected: {inferred}")

    df = df.asfreq(inferred)
    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    y = df["value"].astype("float64")
    if not np.isfinite(y).all():
        raise ValueError("Non-finite values detected in series.")
    if len(y) < 24:
        raise ValueError("Minimum 24 observations required (2 seasonal cycles).")

    # --------------------------------------------------
    # M1: MULTI-START VARIANCE SEARCH
    # --------------------------------------------------

    res, best_llf, best_ls, best_ts = _select_best_bsts(y.values, inferred)

    hist_fitted = res.fittedvalues.astype("float64")
    if hist_fitted.isna().any():
        raise RuntimeError("NaN in fitted values.")

    hist_block = pd.DataFrame({
        "date":      hist_fitted.index,
        "actual":    np.nan,
        "forecast":  hist_fitted.values,
        "ci_low":    np.nan,
        "ci_mid":    hist_fitted.values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------

    forecast_res = res.get_forecast(steps=horizon)
    future_mean  = forecast_res.predicted_mean.astype("float64")
    ci           = forecast_res.conf_int(
                       alpha=1.0 - confidence_level
                   ).astype("float64")

    if future_mean.index.min() <= hist_fitted.index.max():
        raise RuntimeError("Forecast horizon overlaps historical data.")
    if ci.isna().any().any():
        raise RuntimeError("Invalid confidence intervals detected.")
    if not np.isfinite(future_mean).all():
        raise RuntimeError("Non-finite forecast values detected.")
    if (ci.iloc[:, 0] > ci.iloc[:, 1]).any():
        raise RuntimeError("CI bounds inverted.")

    future_block = pd.DataFrame({
        "date":      future_mean.index,
        "actual":    np.nan,
        "forecast":  future_mean.values,
        "ci_low":    ci.iloc[:, 0].values,
        "ci_mid":    future_mean.values,
        "ci_high":   ci.iloc[:, 1].values,
        "error_pct": np.nan,
    })

    for b in (hist_block, future_block):
        b[["forecast","ci_low","ci_mid","ci_high"]] = \
            b[["forecast","ci_low","ci_mid","ci_high"]].astype("float64")

    forecast_df = pd.concat([hist_block, future_block], ignore_index=True)
    if forecast_df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in final output.")
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return ForecastResult(
        model_name  = "BSTS",
        forecast_df = forecast_df[
            ["date","actual","forecast","ci_low","ci_mid","ci_high","error_pct"]
        ],
        metrics  = None,
        metadata = {
            "structure":          "local_linear_trend + seasonal(12)",
            "selection_method":   "multi_start_llf_grid",
            "best_llf":           round(best_llf, 4),
            "best_level_scale":   best_ls,
            "best_trend_scale":   best_ts,
            "n_starts":           len(LEVEL_VAR_SCALES) * len(TREND_VAR_SCALES),
            "bayesian_style":     True,
            "frequency":          inferred,
            "confidence_level":   confidence_level,
            "ci_method":          "state_space_kalman",
            "output_contract":    "ForecastResult",
        },
    )
