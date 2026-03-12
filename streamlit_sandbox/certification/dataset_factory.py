# ==================================================
# FILE: streamlit_sandbox/certification/dataset_factory.py
# ROLE: DETERMINISTIC DATASET FACTORY
# STANDARD: FORTUNE 100 / FIXED SEED / NO DRIFT
# ==================================================

from __future__ import annotations

import numpy as np
import pandas as pd

DETERMINISTIC_SEED = 42
PERIODS = 120
FREQ = "MS"


def _base_index() -> pd.DatetimeIndex:
    return pd.date_range(start="2015-01-01", periods=PERIODS, freq=FREQ)


def seasonal_dataset() -> pd.DataFrame:
    np.random.seed(DETERMINISTIC_SEED)
    idx = _base_index()
    seasonal = 10 + 5 * np.sin(2 * np.pi * np.arange(PERIODS) / 12)
    return pd.DataFrame({"date": idx, "value": seasonal})


def linear_trend_dataset() -> pd.DataFrame:
    np.random.seed(DETERMINISTIC_SEED)
    idx = _base_index()
    trend = np.linspace(10, 200, PERIODS)
    return pd.DataFrame({"date": idx, "value": trend})


def flat_dataset() -> pd.DataFrame:
    np.random.seed(DETERMINISTIC_SEED)
    idx = _base_index()
    flat = np.full(PERIODS, 100.0)
    return pd.DataFrame({"date": idx, "value": flat})


def structural_break_dataset() -> pd.DataFrame:
    np.random.seed(DETERMINISTIC_SEED)
    idx = _base_index()
    base = np.linspace(50, 150, PERIODS)
    spike_start = 60
    base[spike_start:spike_start + 12] *= 1.8
    return pd.DataFrame({"date": idx, "value": base})


def high_noise_dataset() -> pd.DataFrame:
    np.random.seed(DETERMINISTIC_SEED)
    idx = _base_index()
    base = 100 + np.random.normal(0, 20, PERIODS)
    return pd.DataFrame({"date": idx, "value": base})