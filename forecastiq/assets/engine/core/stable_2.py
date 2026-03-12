import numpy as np
import pandas as pd
from datetime import timedelta
import time
from contextlib import contextmanager

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------
# Prophet (OPT-IN ONLY)
# ---------------------------
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# ---------------------------
# Utilities
# ---------------------------

def infer_frequency(idx: pd.Series):
    try:
        return pd.infer_freq(idx)
    except Exception:
        return None


def _get_season_length(freq):
    if freq in ("M", "MS"):
        return 12
    if freq in ("W", "W-SUN"):
        return 52
    if freq in ("D",):
        return 7
    return None


def _build_future_index(last_date, horizon, freq, horizon_unit):
    if horizon_unit == "Periods":
        if not freq:
            raise ValueError("Frequency required for Period-based forecasting")
        return pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    return pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq="D",
    )


def _safe_series(s):
    if s is None:
        return None
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.fillna(method="ffill").fillna(method="bfill")


# ---------------------------
# Model Eligibility
# ---------------------------

def _should_use_prophet(ts: pd.Series, freq: str) -> bool:
    if not PROPHET_AVAILABLE:
        return False
    if freq not in ("M", "MS", "W", "W-SUN"):
        return False
    if len(ts) < 12:
        return False
    return True


# ---------------------------
# ETS
# ---------------------------

def _ets_model_full(ts, freq, horizon, horizon_unit):
    try:
        season_len = _get_season_length(freq)
        if season_len and len(ts) < season_len * 2:
            season_len = None

        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add" if season_len else None,
            seasonal_periods=season_len,
        ).fit()

        fc = model.forecast(horizon)
        future_idx = _build_future_index(ts.index[-1], horizon, freq, horizon_unit)

        resid = ts - model.fittedvalues
        resid_std = resid.std()

        return {
            "forecast": pd.Series(fc.values, index=future_idx),
            "ci_low": pd.Series(fc - 1.96 * resid_std, index=future_idx),
            "ci_high": pd.Series(fc + 1.96 * resid_std, index=future_idx),
            "info": {"model": "ETS"},
            "metrics": {"rmse": float(resid_std)},
        }
    except Exception:
        return None


# ---------------------------
# Linear
# ---------------------------

def _linear_model_full(ts, horizon, horizon_unit):
    try:
        y = ts.values
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)

        future_x = np.arange(len(y), len(y) + horizon)
        fc = coef[0] * future_x + coef[1]

        freq = ts.index.freq or infer_frequency(ts.index.to_series())
        future_idx = _build_future_index(ts.index[-1], horizon, freq, horizon_unit)

        resid = y - (coef[0] * x + coef[1])
        resid_std = resid.std()

        return {
            "forecast": pd.Series(fc, index=future_idx),
            "ci_low": pd.Series(fc - 1.96 * resid_std, index=future_idx),
            "ci_high": pd.Series(fc + 1.96 * resid_std, index=future_idx),
            "info": {"model": "Linear"},
            "metrics": {"rmse": float(resid_std)},
        }
    except Exception:
        return None


# ---------------------------
# Prophet (Windows-safe)
# ---------------------------

def _prophet_model_full(ts, freq, horizon, prophet_cfg: dict):
    if not PROPHET_AVAILABLE or len(ts) < 8:
        return None

    allowed_keys = {
        "growth",
        "seasonality_mode",
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "weekly_seasonality",
        "yearly_seasonality",
        "daily_seasonality",
    }
    safe_cfg = {k: v for k, v in prophet_cfg.items() if k in allowed_keys}

    try:
        df = ts.reset_index()
        df.columns = ["ds", "y"]

        m = Prophet(**safe_cfg)
        m.fit(df)

        future = m.make_future_dataframe(periods=horizon, freq=freq)
        fc = m.predict(future).iloc[-horizon:]

        return {
            "forecast": pd.Series(fc["yhat"].values, index=pd.to_datetime(fc["ds"])),
            "ci_low": pd.Series(fc["yhat_lower"].values, index=pd.to_datetime(fc["ds"])),
            "ci_high": pd.Series(fc["yhat_upper"].values, index=pd.to_datetime(fc["ds"])),
            "info": {"model": "Prophet"},
            "metrics": None,
        }
    except Exception:
        return None


# ---------------------------
# Backtest (Champion–Challenger)
# ---------------------------

def _backtest_rmse(ts, model_fn, freq, k=6):
    if len(ts) <= k + 3:
        return None
    train = ts.iloc[:-k]
    test = ts.iloc[-k:]

    result = model_fn(train, freq, k, "Periods")
    if not result:
        return None

    pred = result["forecast"].iloc[:k]
    return float(np.sqrt(np.mean((test.values - pred.values) ** 2)))


# ---------------------------
# Regime Detection
# ---------------------------

def _detect_regime(ts):
    diffs = ts.diff().dropna()
    if len(diffs) < 5:
        return "None"

    z = (diffs.iloc[-1] - diffs.mean()) / diffs.std()
    if abs(z) > 2.5:
        return "Shock Detected"
    if abs(z) > 1.5:
        return "Volatile"
    return "Stable"


# ---------------------------
# Main Entry
# ---------------------------

def forecast_series(
    ts: pd.Series,
    freq_choice: str,
    horizon: int,
    horizon_unit: str,
    prophet_cfg: dict,
    model_strategy: str,
):
    start_time = time.perf_counter()

    freq = ts.index.freqstr or infer_frequency(ts.index.to_series())
    if freq_choice == "Monthly":
        freq = freq or "MS"
    elif freq_choice == "Weekly":
        freq = freq or "W"

    comparison = {}
    scores = {}

    ets = _ets_model_full(ts, freq, horizon, horizon_unit)
    if ets:
        comparison["ETS"] = ets
        scores["ETS"] = _backtest_rmse(ts, _ets_model_full, freq)

    lin = _linear_model_full(ts, horizon, horizon_unit)
    if lin:
        comparison["Linear"] = lin
        scores["Linear"] = _backtest_rmse(ts, lambda a, f, h, u: _linear_model_full(a, h, u), freq)

    prop = None
    if model_strategy in ("Auto", "Prophet") and _should_use_prophet(ts, freq):
        prop = _prophet_model_full(ts, freq, horizon, prophet_cfg)
        if prop:
            comparison["Prophet"] = prop
            scores["Prophet"] = None  # Prophet backtest optional later

    # Champion selection
    winner = min(
        ((k, v) for k, v in scores.items() if v is not None),
        key=lambda x: x[1],
        default=("ETS", None),
    )[0]

    selected = comparison[winner]

    # Decision metadata
    trend = ts.diff().mean()
    trend_strength = "Strong" if abs(trend) > ts.std() else "Moderate"
    risk = _detect_regime(ts)

    audit = {
        "requested_model": model_strategy,
        "selected_model": winner,
        "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
        "model_scores": scores,
        "selection_rule": "champion_challenger_rmse",
        "risk_flag": risk,
    }

    ui_meta = {
        "forecast_confidence": "High" if risk == "Stable" else "Moderate",
        "trend_strength": trend_strength,
        "risk_flag": risk,
    }

    return (
        _safe_series(selected["forecast"]),
        _safe_series(selected["ci_low"]),
        _safe_series(selected["ci_high"]),
        selected["info"],
        selected["metrics"],
        comparison,
        audit,
        ui_meta,
    )
