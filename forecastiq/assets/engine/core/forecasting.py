import numpy as np
import pandas as pd
import time
import math

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ============================================================
# Optional Prophet (OPT-IN ONLY)
# ============================================================

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# ============================================================
# Frequency Normalization (ENGINE-AUTHORITATIVE)
# ============================================================

FREQ_ALIASES = {
    "MONTHLY": "MS",
    "MONTH": "MS",
    "MON": "MS",
    "M": "MS",
    "MS": "MS",

    "WEEKLY": "W",
    "WEEK": "W",
    "WK": "W",
    "W": "W",

    "DAILY": "D",
    "DAY": "D",
    "D": "D",
}


def normalize_frequency(freq_choice: str | None, ts_freq: str | None) -> str:
    if freq_choice:
        key = str(freq_choice).strip().upper()
        if key in FREQ_ALIASES:
            return FREQ_ALIASES[key]
        raise ValueError(f"Invalid frequency: {freq_choice}")

    # AUTO-INFER (NEW — preserves prior behavior)
    if ts_freq:
        return ts_freq

    raise ValueError("Unable to determine frequency for forecasting")


# ============================================================
# Utilities
# ============================================================

def _ensure_datetime_index(ts: pd.Series) -> pd.Series:
    ts = ts.copy()
    if not isinstance(ts.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        ts.index = pd.to_datetime(ts.index)

    if isinstance(ts.index, pd.PeriodIndex):
        ts.index = ts.index.to_timestamp()

    return ts.sort_index()


def _safe_series(ts: pd.Series) -> pd.Series:
    if ts is None or len(ts) == 0:
        raise ValueError("Empty series")

    ts = _ensure_datetime_index(ts)
    ts = ts.replace([np.inf, -np.inf], np.nan)
    ts = pd.to_numeric(ts, errors="coerce").astype(float)
    ts = ts.dropna()

    if len(ts) == 0:
        raise ValueError("Series has no valid numeric values")

    return ts


def _season_length(freq: str):
    return {"MS": 12, "W": 52, "D": 7}.get(freq)


def _future_index(last_date, periods, freq):
    return pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=periods,
        freq=freq,
    )


def _coerce_horizon_to_periods(horizon: int, horizon_unit: str, freq: str) -> int:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be > 0")

    unit = (horizon_unit or "").lower()

    if unit in ("", "period", "periods"):
        return h

    if unit in ("months", "month"):
        return h if freq == "MS" else max(1, int(round(h * 4)))

    if unit in ("weeks", "week"):
        return h if freq == "W" else max(1, int(round(h * 7)))

    if unit in ("days", "day"):
        return h if freq == "D" else max(1, int(round(h / 30)))

    raise ValueError(f"Invalid horizon_unit: {horizon_unit}")


def _align_and_impute(ts: pd.Series, freq: str):
    full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq=freq)
    aligned = ts.reindex(full_idx)

    missing = aligned.isna().sum()
    total = len(aligned)

    if missing > 0:
        aligned = aligned.interpolate(method="time").ffill().bfill()

    if aligned.isna().any():
        raise ValueError("Series could not be imputed to continuous frequency")

    return aligned, {
        "aligned_frequency": freq,
        "missing_points": int(missing),
        "total_points": int(total),
        "missing_ratio": float(missing / total),
        "imputation_method": "linear" if missing else None,
    }


# ============================================================
# Models
# ============================================================

def _ets_model(ts, freq, periods):
    season_len = _season_length(freq)
    if season_len and len(ts) < season_len * 2:
        season_len = None

    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add" if season_len else None,
        seasonal_periods=season_len,
    ).fit()

    forecast = model.forecast(periods)
    resid = ts - model.fittedvalues
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    idx = _future_index(ts.index[-1], periods, freq)
    fc = pd.Series(forecast.values, index=idx)

    return {
        "forecast": fc,
        "ci_low": fc - 1.96 * rmse,
        "ci_high": fc + 1.96 * rmse,
        "info": {"model": "ETS"},
        "metrics": {"rmse": rmse},
    }


def _linear_model(ts, freq, periods):
    y = ts.values
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    resid = y - (coef[0] * x + coef[1])
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    future_x = np.arange(len(y), len(y) + periods)
    idx = _future_index(ts.index[-1], periods, freq)
    fc = pd.Series(coef[0] * future_x + coef[1], index=idx)

    return {
        "forecast": fc,
        "ci_low": fc - 1.96 * rmse,
        "ci_high": fc + 1.96 * rmse,
        "info": {"model": "Linear"},
        "metrics": {"rmse": rmse},
    }


def _prophet_model(ts, freq, periods, cfg):
    if not PROPHET_AVAILABLE:
        return None

    df = ts.reset_index()
    df.columns = ["ds", "y"]

    m = Prophet(**cfg)
    m.fit(df)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future).iloc[-periods:]

    return {
        "forecast": pd.Series(fc["yhat"].values, index=pd.to_datetime(fc["ds"])),
        "ci_low": pd.Series(fc["yhat_lower"].values, index=pd.to_datetime(fc["ds"])),
        "ci_high": pd.Series(fc["yhat_upper"].values, index=pd.to_datetime(fc["ds"])),
        "info": {"model": "Prophet"},
        "metrics": None,
    }


# ============================================================
# Auto Model Selection (HARDENED)
# ============================================================

def _select_model(results, freq, periods):
    rmses = {
        k: v["metrics"]["rmse"]
        for k, v in results.items()
        if v and v.get("metrics") and v["metrics"].get("rmse") is not None
    }

    if not rmses:
        return "ETS", "Fallback: no RMSE-capable models"

    best = min(rmses, key=rmses.get)
    tolerance = 0.02

    close = [
        k for k, r in rmses.items()
        if abs(r - rmses[best]) / rmses[best] <= tolerance
    ]

    if "ETS" in close and freq == "MS" and periods > 1:
        return "ETS", "Structural preference: season-capable model"

    return best, "RMSE-dominant"


# ============================================================
# Main Engine Entry
# ============================================================

def forecast_series(ts, freq_choice, horizon, horizon_unit, prophet_cfg, model_strategy):
    start = time.perf_counter()

    ts = _safe_series(ts)
    freq = normalize_frequency(freq_choice, ts.index.freqstr)
    periods = _coerce_horizon_to_periods(horizon, horizon_unit, freq)
    ts_aligned, quality = _align_and_impute(ts, freq)

    comparison = {
        "ETS": _ets_model(ts_aligned, freq, periods),
        "Linear": _linear_model(ts_aligned, freq, periods),
    }

    if model_strategy.lower() in ("auto", "prophet"):
        p = _prophet_model(ts_aligned, freq, periods, prophet_cfg or {})
        if p:
            comparison["Prophet"] = p

    winner, rationale = _select_model(comparison, freq, periods)
    selected = comparison[winner]

    audit = {
        "requested_model": model_strategy,
        "selected_model": winner,
        "selection_rationale": rationale,
        "execution_time_ms": int((time.perf_counter() - start) * 1000),
        "risk_flag": "Normal",
        "data_quality": quality,
        "effective_periods": periods,
        "effective_frequency": freq,
    }

    return (
        selected["forecast"],
        selected["ci_low"],
        selected["ci_high"],
        selected["info"],
        selected["metrics"],
        comparison,
        audit,
    )
