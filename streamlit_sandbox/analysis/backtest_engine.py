# ==================================================
# FILE: analysis/backtest_engine.py
# ROLE: PRODUCTION ROLLING-ORIGIN CERTIFICATION ENGINE
# STATUS: EXECUTIVE-GRADE / DETERMINISTIC / SEASONALLY-CORRECT / ROBUST
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any

MIN_OBSERVATIONS = 36
ROLLING_FOLDS = 3
SEASONAL_PERIOD = 12  # Monthly seasonality


# ==================================================
# CORE METRIC FUNCTIONS
# ==================================================

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def _bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def _directional_accuracy(y_true, y_pred, y_train_last):
    """
    Percent of periods where model predicted correct direction of change.
    """
    true_diff = np.sign(y_true - y_train_last)
    pred_diff = np.sign(y_pred - y_train_last)
    return float(np.mean(true_diff == pred_diff))


def _mase(y_true, y_pred, y_train):
    """
    Seasonal MASE using seasonal naïve scaling.
    Falls back to non-seasonal if insufficient length.
    """
    if len(y_train) > SEASONAL_PERIOD:
        naive_errors = np.abs(
            y_train[SEASONAL_PERIOD:] - y_train[:-SEASONAL_PERIOD]
        )
    else:
        naive_errors = np.abs(np.diff(y_train))

    scale = np.mean(naive_errors)

    if scale == 0 or np.isnan(scale):
        return np.nan

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _theils_u(y_true, y_pred, y_train):
    """
    Theil's U relative to rolling naïve forecast.
    """
    naive_forecast = []
    last_value = y_train[-1]

    for actual in y_true:
        naive_forecast.append(last_value)
        last_value = actual

    naive_forecast = np.array(naive_forecast)

    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denominator = np.sqrt(np.mean((y_true - naive_forecast) ** 2))

    if denominator == 0 or np.isnan(denominator):
        return np.nan

    return float(numerator / denominator)


def _ci_coverage(y_true, ci_low, ci_high):
    inside = (y_true >= ci_low) & (y_true <= ci_high)
    return float(np.mean(inside))


# ==================================================
# MAIN CERTIFICATION ENGINE
# ==================================================

def run_backtest(
    df: pd.DataFrame,
    model_runner: Callable,
    horizon: int,
    confidence_level: float,
) -> Dict[str, Any]:

    series = df.copy()
    series = series.sort_values("date")

    if len(series) < MIN_OBSERVATIONS:
        return {
            "eligible": False,
            "reason": f"Minimum {MIN_OBSERVATIONS} observations required for certification.",
        }

    y_full = series["value"].astype(float).values
    fold_metrics = []

    for fold in range(ROLLING_FOLDS):

        train_end = len(y_full) - horizon - (ROLLING_FOLDS - fold - 1) * horizon
        train_df = series.iloc[:train_end].copy()
        test_df = series.iloc[train_end:train_end + horizon].copy()

        if len(test_df) < horizon:
            continue

        result = model_runner(
            train_df,
            horizon=horizon,
            confidence_level=confidence_level,
        )

        forecast_df = result.forecast_df.copy()
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        last_train_date = pd.to_datetime(train_df["date"]).max()

        future_block = forecast_df[
            forecast_df["date"] > last_train_date
        ].reset_index(drop=True)

        if len(future_block) != horizon:
            continue

        y_true = test_df["value"].values
        y_pred = future_block["forecast"].values

        # CI guard (no crash if missing)
        if {"ci_low", "ci_high"}.issubset(future_block.columns):
            ci_low = future_block["ci_low"].values
            ci_high = future_block["ci_high"].values
            coverage = _ci_coverage(y_true, ci_low, ci_high)
        else:
            coverage = np.nan

        fold_metrics.append({
            "mae": _mae(y_true, y_pred),
            "rmse": _rmse(y_true, y_pred),
            "mape": _mape(y_true, y_pred),
            "smape": _smape(y_true, y_pred),
            "bias": _bias(y_true, y_pred),
            "mase": _mase(y_true, y_pred, train_df["value"].values),
            "theils_u": _theils_u(y_true, y_pred, train_df["value"].values),
            "ci_coverage": coverage,
            "directional_accuracy": _directional_accuracy(
                y_true,
                y_pred,
                train_df["value"].values[-1],
            ),
        })

    if not fold_metrics:
        return {
            "eligible": False,
            "reason": "Insufficient fold generation for certification.",
        }

    aggregated = {}

    for key in fold_metrics[0].keys():
        aggregated[key] = float(
            np.nanmean([m[key] for m in fold_metrics])
        )

    # Fold stability metrics (variance across folds)
    for key in fold_metrics[0].keys():
        aggregated[f"{key}_std"] = float(
            np.nanstd([m[key] for m in fold_metrics])
        )

    aggregated["eligible"] = True
    aggregated["folds"] = len(fold_metrics)
    aggregated["mean_level"] = float(np.mean(series["value"]))

    return aggregated
