# ==================================================
# FILE: sentinel_engine/models/lightgbm_model.py
# VERSION: 2.0.0
# MODEL: LIGHTGBM WITH LAG + EXOGENOUS FEATURES
# ENGINE: Sentinel Engine v2.0.0
# TIER: pro (minimum)
# STATUS: VEDUTA ENGINE — PHASE 3C
# ==================================================
#
# PURPOSE:
#   Gradient boosted tree model for time series forecasting.
#   Captures complex non-linear feature interactions that
#   linear models cannot model.
#
#   Strongest on: promotional lift, pricing effects, macro
#   variable interactions, complex seasonal patterns with
#   external drivers.
#
# FEATURE SET:
#   Lag features:     y(t-1), y(t-2), y(t-3), y(t-6), y(t-12)
#   Rolling stats:    rolling mean and std over 3 and 6 periods
#   Calendar:         month, quarter, week-of-year
#   Exogenous:        any additional numeric columns in df
#
# EXOGENOUS INPUTS:
#   Pass extra numeric columns in df alongside 'date' and 'value'.
#   At forecast time, exogenous columns must cover the full
#   horizon (future rows appended to df with future exog values).
#   If no exogenous columns present, model runs on lag features only.
#   Falls back to lag-only mode automatically if exog missing at
#   forecast time.
#
# MULTI-STEP FORECAST:
#   Recursive strategy: predicted y(t+1) used as lag feature
#   for y(t+2), etc. Standard for tree-based TS models.
#
# CI METHOD:
#   Quantile regression: LightGBM trained at alpha/2 and
#   1-alpha/2 quantiles for lower and upper bounds.
#   Three models: point (mse), lower (quantile), upper (quantile).
#
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from sentinel_engine.models.contracts import ForecastResult

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

LAG_COLS    = [1, 2, 3, 6, 12]
ROLL_WINS   = [3, 6]
LGB_PARAMS_POINT = {
    "objective":        "regression",
    "metric":           "rmse",
    "n_estimators":     200,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples":5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "verbose":          -1,
    "random_state":     42,
}


def _lgb_params_quantile(alpha: float) -> dict:
    return {
        "objective":        "quantile",
        "alpha":            alpha,
        "metric":           "quantile",
        "n_estimators":     200,
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_child_samples":5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "verbose":          -1,
        "random_state":     42,
    }


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def _build_features(
    df:       pd.DataFrame,
    lag_cols: list[int],
    roll_win: list[int],
    exog_cols: list[str],
) -> pd.DataFrame:
    """
    Build lag, rolling, calendar, and exogenous features.
    Returns feature DataFrame aligned with df index.
    """
    feat = pd.DataFrame(index=df.index)

    # Lag features
    for lag in lag_cols:
        feat[f"lag_{lag}"] = df["value"].shift(lag)

    # Rolling features
    for w in roll_win:
        feat[f"roll_mean_{w}"] = df["value"].shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"]  = df["value"].shift(1).rolling(w).std()

    # Calendar features
    feat["month"]   = df["date"].dt.month.astype("float64")
    feat["quarter"] = df["date"].dt.quarter.astype("float64")

    # Exogenous features
    for col in exog_cols:
        feat[col] = df[col].astype("float64")

    return feat


def _get_exog_cols(df: pd.DataFrame) -> list[str]:
    """Return exogenous column names (all except 'date' and 'value')."""
    return [c for c in df.columns if c not in ("date", "value")]


# ==================================================
# MODEL RUNNER
# ==================================================

def run_lightgbm(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
) -> ForecastResult:

    # --------------------------------------------------
    # STRICT INPUT VALIDATION
    # --------------------------------------------------

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("LightGBM requires 'date' and 'value' columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred is None:
        raise ValueError("Frequency cannot be inferred.")

    if df["value"].isna().any():
        raise ValueError("Missing values in 'value' column.")

    if not np.isfinite(df["value"].astype("float64").values).all():
        raise ValueError("Non-finite values in series.")

    min_obs = max(LAG_COLS) + max(ROLL_WINS) + 2
    if len(df) < min_obs:
        raise ValueError(f"Minimum {min_obs} observations required.")

    # --------------------------------------------------
    # EXOGENOUS COLUMN DETECTION
    # --------------------------------------------------

    exog_cols    = _get_exog_cols(df)
    exog_enabled = len(exog_cols) > 0

    # Validate exog columns are numeric
    for col in exog_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Exogenous column '{col}' must be numeric.")

    # --------------------------------------------------
    # FUTURE DATE INDEX
    # --------------------------------------------------

    last_date  = df["date"].iloc[-1]
    freq_alias = inferred if inferred else "MS"
    future_idx = pd.date_range(
        start=last_date,
        periods=horizon + 1,
        freq=freq_alias
    )[1:]

    if len(future_idx) != horizon:
        raise RuntimeError(f"Future date index mismatch: {len(future_idx)} vs {horizon}.")

    # --------------------------------------------------
    # CHECK EXOG COVERAGE AT FORECAST HORIZON
    # --------------------------------------------------

    # If exog enabled, future exog rows must exist in df
    # (user appends future dates + exog values to df before calling)
    future_exog_available = False
    if exog_enabled:
        df_future_rows = df[df["date"].isin(future_idx)]
        if len(df_future_rows) == horizon:
            future_exog_available = True
        else:
            # Fallback: lag-only mode
            exog_cols    = []
            exog_enabled = False

    # --------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------

    feat = _build_features(df, LAG_COLS, ROLL_WINS, exog_cols)

    # Training rows: history only, drop NaN from lag construction
    hist_mask  = df["value"].notna()
    feat_train = feat[hist_mask].copy()
    y_train    = df.loc[hist_mask, "value"].astype("float64").values

    # Drop rows with NaN features (lag warmup)
    valid_mask   = feat_train.notna().all(axis=1)
    feat_train   = feat_train[valid_mask]
    y_train      = y_train[valid_mask]
    feature_cols = feat_train.columns.tolist()

    if len(feat_train) < 10:
        raise ValueError("Insufficient training rows after lag construction.")

    X_train = feat_train[feature_cols].values.astype("float64")

    # --------------------------------------------------
    # TRAIN THREE MODELS: POINT + LOWER + UPPER
    # --------------------------------------------------

    alpha_ci  = 1.0 - confidence_level
    q_lo      = alpha_ci / 2.0
    q_hi      = 1.0 - alpha_ci / 2.0

    model_point = lgb.LGBMRegressor(**LGB_PARAMS_POINT)
    model_lo    = lgb.LGBMRegressor(**_lgb_params_quantile(q_lo))
    model_hi    = lgb.LGBMRegressor(**_lgb_params_quantile(q_hi))

    try:
        model_point.fit(X_train, y_train)
        model_lo.fit(X_train, y_train)
        model_hi.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"LightGBM training failed: {e}") from e

    # --------------------------------------------------
    # HISTORICAL FITTED VALUES
    # --------------------------------------------------

    fitted_values = np.full(len(df), np.nan)
    train_indices = feat_train.index.tolist()
    fitted_values[train_indices] = model_point.predict(X_train).astype("float64")

    hist_block = pd.DataFrame({
        "date":      df["date"].values,
        "actual":    np.nan,
        "forecast":  fitted_values,
        "ci_low":    np.nan,
        "ci_mid":    fitted_values,
        "ci_high":   np.nan,
        "error_pct": np.nan,
    })

    # --------------------------------------------------
    # RECURSIVE MULTI-STEP FORECAST
    # --------------------------------------------------

    # Extend df with future rows for feature construction
    if exog_enabled and future_exog_available:
        df_future = df[df["date"].isin(future_idx)].copy()
        df_future["value"] = np.nan
    else:
        df_future = pd.DataFrame({
            "date":  future_idx,
            "value": np.nan,
        })

    df_extended = pd.concat([df, df_future], ignore_index=True)
    df_extended = df_extended.sort_values("date").reset_index(drop=True)

    point_preds = []
    lo_preds    = []
    hi_preds    = []

    hist_len = len(df)

    for step in range(horizon):
        feat_ext = _build_features(df_extended, LAG_COLS, ROLL_WINS, exog_cols)
        row_idx  = hist_len + step
        x_row    = feat_ext.loc[row_idx, feature_cols].values.astype("float64")

        if not np.isfinite(x_row).all():
            # Use last valid prediction as fallback
            fallback = point_preds[-1] if point_preds else float(df["value"].iloc[-1])
            point_preds.append(fallback)
            lo_preds.append(fallback * 0.9)
            hi_preds.append(fallback * 1.1)
        else:
            x_2d = x_row.reshape(1, -1)
            p    = float(model_point.predict(x_2d)[0])
            lo   = float(model_lo.predict(x_2d)[0])
            hi   = float(model_hi.predict(x_2d)[0])

            point_preds.append(p)
            lo_preds.append(lo)
            hi_preds.append(hi)

        # Feed prediction back into extended df
        df_extended.at[row_idx, "value"] = point_preds[-1]

    point_arr = np.array(point_preds, dtype="float64")
    lo_arr    = np.array(lo_preds,    dtype="float64")
    hi_arr    = np.array(hi_preds,    dtype="float64")

    # Ensure CI brackets point forecast
    lo_arr = np.minimum(lo_arr, point_arr)
    hi_arr = np.maximum(hi_arr, point_arr)

    if not np.isfinite(point_arr).all():
        raise RuntimeError("Non-finite values in point forecast.")

    # --------------------------------------------------
    # FUTURE BLOCK
    # --------------------------------------------------

    future_block = pd.DataFrame({
        "date":      future_idx,
        "actual":    np.nan,
        "forecast":  point_arr,
        "ci_low":    lo_arr,
        "ci_mid":    point_arr,
        "ci_high":   hi_arr,
        "error_pct": np.nan,
    })

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

    future_rows = forecast_df.tail(horizon)
    if future_rows[["ci_low", "ci_high"]].isna().any().any():
        raise RuntimeError("NaN CI in future forecast output.")
    if (future_rows["ci_low"] > future_rows["ci_high"]).any():
        raise RuntimeError("Inverted CI in future forecast output.")

    return ForecastResult(
        model_name  = "LightGBM",
        forecast_df = forecast_df[
            ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
        ],
        metrics  = None,
        metadata = {
            "feature_cols":          feature_cols,
            "lag_cols":              LAG_COLS,
            "roll_windows":          ROLL_WINS,
            "exog_cols":             exog_cols,
            "exog_enabled":          exog_enabled,
            "future_exog_available": future_exog_available if exog_enabled else False,
            "ci_method":             "quantile_regression",
            "q_lo":                  round(q_lo, 4),
            "q_hi":                  round(q_hi, 4),
            "n_estimators":          LGB_PARAMS_POINT["n_estimators"],
            "frequency":             inferred,
            "confidence_level":      confidence_level,
            "min_tier":              "pro",
            "output_contract":            "ForecastResult",
        },
    )
