# ==================================================
# FILE: veduta_project/data/hostile_series.py
# ROLE: HOSTILE SERIES GENERATOR
# PURPOSE: Generate 6 challenging synthetic time series
#          to smoke-test the engine before M3 simulation.
# USAGE: python data/hostile_series.py
#        Writes 6 CSV files to data/hostile/
# GOVERNANCE: Read-only relative to sentinel_engine/.
#             No engine files are modified.
# ==================================================

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from datetime import date

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "hostile")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)

START_DATE = "2015-01-01"


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=START_DATE, periods=n, freq="MS")


def _save(name: str, dates: pd.DatetimeIndex, values: np.ndarray) -> str:
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": values})
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def series_01_flat() -> str:
    """
    HOSTILE SERIES 1 — FLAT / NEAR-ZERO VARIANCE
    A constant series with tiny noise.
    Danger: MASE denominator collapses toward zero.
    Models that divide by near-zero scale will produce
    infinite or NaN MASE. Tests G7 floor guard in backtest.py.
    """
    n      = 72
    dates  = _make_dates(n)
    values = np.full(n, 1000.0) + rng.normal(0, 0.5, n)
    return _save("01_flat", dates, values)


def series_02_structural_break() -> str:
    """
    HOSTILE SERIES 2 — STRUCTURAL BREAK (REGIME CHANGE)
    Series runs normally for 48 months then jumps 60% permanently.
    Danger: Models trained on pre-break data will systematically
    under-forecast post-break. Tests regime-change robustness.
    Prophet was already flagged for this pattern in certification.
    """
    n      = 72
    dates  = _make_dates(n)
    trend  = np.linspace(500, 700, n)
    seasonal = 80 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise  = rng.normal(0, 20, n)
    values = trend + seasonal + noise
    # Break at month 48: permanent +400 level shift
    values[48:] += 400
    return _save("02_structural_break", dates, values)


def series_03_short() -> str:
    """
    HOSTILE SERIES 3 — SHORT SERIES (48 OBSERVATIONS)
    Minimum viable length for monthly backtest with 3 folds.
    Danger: Models requiring long initialization (TBATS, BSTS,
    DHR) may fail or produce degenerate results. Tests
    MIN_OBSERVATIONS guard and fold generation logic.
    """
    n      = 48
    dates  = _make_dates(n)
    trend  = np.linspace(200, 400, n)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise  = rng.normal(0, 15, n)
    values = trend + seasonal + noise
    return _save("03_short", dates, values)


def series_04_high_volatility() -> str:
    """
    HOSTILE SERIES 4 — HIGH VOLATILITY / NOISY
    Signal-to-noise ratio is very low. Random shocks dominate.
    Danger: ML models (LightGBM, NNETAR) may overfit to noise.
    GARCH should perform relatively better here.
    Tests ensemble stability on unpredictable series.
    """
    n      = 72
    dates  = _make_dates(n)
    trend  = np.linspace(1000, 1200, n)
    seasonal = 100 * np.sin(2 * np.pi * np.arange(n) / 12)
    # Noise std is 3x the seasonal amplitude
    noise  = rng.normal(0, 300, n)
    values = np.maximum(trend + seasonal + noise, 0)
    return _save("04_high_volatility", dates, values)


def series_05_intermittent() -> str:
    """
    HOSTILE SERIES 5 — INTERMITTENT DEMAND (40% ZEROS)
    Frequent zero periods simulate spare parts / seasonal SKUs.
    Danger: Standard models (ETS, SARIMA) produce negative
    forecasts or collapse. Croston_SBA routing should activate.
    Tests intermittent detection logic in ensemble.py.
    """
    n      = 72
    dates  = _make_dates(n)
    base   = rng.integers(50, 300, n).astype(float)
    # Zero out ~40% of periods randomly
    zero_mask = rng.random(n) < 0.40
    base[zero_mask] = 0.0
    return _save("05_intermittent", dates, base)


def series_06_trend_reversal() -> str:
    """
    HOSTILE SERIES 6 — STRONG TREND THEN REVERSAL
    Upward trend for 48 months, then downward trend for 24.
    Danger: Models that extrapolate trend (Theta, ETS with
    additive trend) will forecast continued growth into a
    declining period. Tests trend-dampening robustness.
    HW_Damped should outperform standard ETS here.
    """
    n      = 72
    dates  = _make_dates(n)
    trend  = np.concatenate([
        np.linspace(500, 1200, 48),
        np.linspace(1200, 800, 24),
    ])
    seasonal = 120 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise  = rng.normal(0, 30, n)
    values = trend + seasonal + noise
    return _save("06_trend_reversal", dates, values)


if __name__ == "__main__":
    generators = [
        series_01_flat,
        series_02_structural_break,
        series_03_short,
        series_04_high_volatility,
        series_05_intermittent,
        series_06_trend_reversal,
    ]

    print("\n── Hostile Series Generator ─────────────────────────────")
    print(f"   Output directory: {OUTPUT_DIR}\n")

    for gen in generators:
        path = gen()
        df   = pd.read_csv(path)
        zeros = (df["value"] == 0).sum()
        print(f"   ✅  {os.path.basename(path):35s} "
              f"{len(df):3d} obs  "
              f"mean={df['value'].mean():8.1f}  "
              f"zeros={zeros}")

    print(f"\n   {len(generators)} series written to {OUTPUT_DIR}")
    print("── Done ─────────────────────────────────────────────────\n")
