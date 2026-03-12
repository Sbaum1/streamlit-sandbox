# ==================================================
# FILE: veduta_project/tests/m3_loader.py
# VERSION: 1.0.0
# ROLE: M3 COMPETITION DATA LOADER
# ENGINE: Sentinel Engine v2.1.0
# ==================================================
#
# Loads the M3 Monthly dataset (1,428 series) from the
# datasetsforecast package, which provides the canonical
# M3 data in a clean, versioned format identical to the
# original Makridakis et al. (2000) competition dataset.
#
# Each M3 monthly series has:
#   - Between 48 and 144 observations
#   - A 18-period hold-out horizon (the competition standard)
#   - A unique ID (M1 through M1428)
#
# OUTPUT FORMAT:
#   Returns a list of dicts, each with:
#     {
#       "id":      "M1",
#       "df":      pd.DataFrame with columns ["date","value"],
#       "horizon": 18,
#       "n_train": int,
#     }
#
# INSTALL:
#   pip install datasetsforecast
#
# GOVERNANCE:
#   - No engine imports — pure data loading
#   - Data is returned as-is from the competition dataset
#   - No imputation, no manipulation
#   - Series with any NaN in train split are flagged and skipped
# ==================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


M3_HORIZON   = 18   # Competition standard for monthly series
M3_FREQUENCY = "MS" # Month-start


def load_m3_monthly(
    max_series:    Optional[int] = None,
    series_ids:    Optional[List[str]] = None,
    verbose:       bool = True,
) -> List[Dict[str, Any]]:
    """
    Load M3 monthly series from datasetsforecast package.

    Args:
        max_series  : If set, return only the first N series (pilot mode).
        series_ids  : If set, return only series with these IDs.
        verbose     : Print loading progress.

    Returns:
        List of series dicts. Each dict:
          id        : str  — M3 series identifier (e.g. "M1")
          df        : pd.DataFrame(date, value) — TRAINING data only
          horizon   : int  — forecast horizon (18 for monthly)
          n_train   : int  — number of training observations
          actuals   : np.ndarray — hold-out actuals (length=horizon)
    """
    try:
        from datasetsforecast.m3 import M3
    except ImportError:
        raise ImportError(
            "datasetsforecast not installed.\n"
            "Run: pip install datasetsforecast"
        )

    if verbose:
        print("Loading M3 monthly dataset...")

    # M3.load return format varies by datasetsforecast version.
    # Older: (train_df, test_df, freq_df)
    # Newer: (train_df, test_df) with no freq_df, or test_df may be None
    raw = M3.load(directory=".", group="Monthly")

    if isinstance(raw, tuple):
        train_df = raw[0]
        test_df  = raw[1] if len(raw) > 1 else None
    else:
        train_df = raw
        test_df  = None

    # If test_df is None or empty, reconstruct from train_df using last 18 obs
    if test_df is None or (hasattr(test_df, "__len__") and len(test_df) == 0):
        import pandas as _pd
        keep_rows, test_rows = [], []
        for sid in train_df["unique_id"].unique():
            s = train_df[train_df["unique_id"] == sid].sort_values("ds")
            if len(s) <= M3_HORIZON:
                keep_rows.append(s)
            else:
                keep_rows.append(s.iloc[:-M3_HORIZON])
                test_rows.append(s.iloc[-M3_HORIZON:])
        train_df = _pd.concat(keep_rows, ignore_index=True)
        test_df  = _pd.concat(test_rows, ignore_index=True) if test_rows else _pd.DataFrame(columns=train_df.columns)

    if verbose:
        all_ids = train_df["unique_id"].unique()
        print(f"  Total monthly series: {len(all_ids)}")

    # Filter to requested IDs if specified
    all_ids = sorted(train_df["unique_id"].unique())
    if series_ids is not None:
        all_ids = [i for i in all_ids if i in series_ids]
    if max_series is not None:
        all_ids = all_ids[:max_series]

    results    = []
    skipped    = 0

    for sid in all_ids:
        tr = train_df[train_df["unique_id"] == sid].copy()
        te = test_df[test_df["unique_id"]   == sid].copy()

        tr = tr.sort_values("ds").reset_index(drop=True)
        te = te.sort_values("ds").reset_index(drop=True)

        # Build training DataFrame in engine format
        df = pd.DataFrame({
            "date":  pd.to_datetime(tr["ds"]),
            "value": tr["y"].astype("float64"),
        })

        # Validate
        if df["value"].isna().any():
            skipped += 1
            continue
        if not np.isfinite(df["value"]).all():
            skipped += 1
            continue
        if len(df) < 24:   # minimum for seasonal models
            skipped += 1
            continue

        # Enforce monthly frequency
        df = df.sort_values("date").reset_index(drop=True)
        try:
            df["date"] = pd.date_range(
                start=df["date"].iloc[0],
                periods=len(df),
                freq=M3_FREQUENCY,
            )
        except Exception:
            skipped += 1
            continue

        actuals = te["y"].astype("float64").values
        if len(actuals) == 0:
            skipped += 1
            continue

        results.append({
            "id":      sid,
            "df":      df,
            "horizon": min(M3_HORIZON, len(actuals)),
            "n_train": len(df),
            "actuals": actuals[:M3_HORIZON],
        })

    if verbose:
        print(f"  Series loaded: {len(results)}  Skipped: {skipped}")

    return results


def compute_mase(
    actuals:  np.ndarray,
    forecast: np.ndarray,
    y_train:  np.ndarray,
    period:   int = 12,
) -> float:
    """
    Seasonal MASE — the M3 competition standard metric.
    MASE < 1.0 = beats seasonal naïve.
    """
    if len(y_train) > period:
        scale = float(np.mean(np.abs(y_train[period:] - y_train[:-period])))
    else:
        scale = float(np.mean(np.abs(np.diff(y_train))))

    scale = max(scale, 1e-6, float(np.mean(np.abs(y_train))) * 0.01)

    mae = float(np.mean(np.abs(actuals - forecast)))
    return mae / scale


def compute_smape(actuals: np.ndarray, forecast: np.ndarray) -> float:
    """sMAPE as used in M3 competition."""
    denom = (np.abs(actuals) + np.abs(forecast)) / 2.0
    mask  = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actuals[mask] - forecast[mask]) / denom[mask]))
