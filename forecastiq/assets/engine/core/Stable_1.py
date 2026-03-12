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
    # Monthly
    "MONTHLY": "MS",
    "MONTH": "MS",
    "MON": "MS",
    "M": "MS",
    "MS": "MS",

    # Weekly
    "WEEKLY": "W",
    "WEEK": "W",
    "WK": "W",
    "W": "W",

    # Daily
    "DAILY": "D",
    "DAY": "D",
    "D": "D",
}


def normalize_frequency(freq_choice: str | None, ts_freq: str | None) -> str:
    """
    Single source of truth for frequency handling.

    Returns a pandas-compatible frequency string:
      - Monthly -> "MS"
      - Weekly  -> "W"
      - Daily   -> "D"
    """
    if freq_choice:
        key = str(freq_choice).strip().upper()
        if key in FREQ_ALIASES:
            return FREQ_ALIASES[key]
        raise ValueError(f"Invalid frequency: {freq_choice}")

    if ts_freq:
        # ts_freq might be "MS", "W", "D" (or something compatible)
        return str(ts_freq)

    raise ValueError("Unable to determine frequency for forecasting")


# ============================================================
# Utilities
# ============================================================

def _sanitize_numeric(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return None


def _ensure_datetime_index(ts: pd.Series) -> pd.Series:
    ts = ts.copy()
    if not isinstance(ts.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            ts.index = pd.to_datetime(ts.index)
        except Exception as e:
            raise ValueError(f"Series index must be datetime-like. Failed to parse index: {e}")

    if isinstance(ts.index, pd.PeriodIndex):
        ts.index = ts.index.to_timestamp()

    # Sort and drop duplicate timestamps (keep last observation)
    ts = ts[~ts.index.duplicated(keep="last")].sort_index()
    return ts


def _safe_series(ts: pd.Series) -> pd.Series:
    """
    Strict sanitation:
    - enforce datetime index
    - replace inf with nan
    - coerce to float
    - DO NOT require continuity here (handled later)
    - drop rows with non-numeric values
    """
    if ts is None or len(ts) == 0:
        raise ValueError("Empty series")

    ts = _ensure_datetime_index(ts)

    ts = ts.copy()
    ts = ts.replace([np.inf, -np.inf], np.nan)

    # Coerce to numeric floats safely
    ts = pd.to_numeric(ts, errors="coerce").astype(float)

    # Drop NaN (we'll handle gaps after frequency alignment)
    ts = ts.dropna()

    if len(ts) == 0:
        raise ValueError("Series has no valid numeric values after sanitation")

    return ts


def _season_length(freq: str):
    if freq == "MS":
        return 12
    if freq == "W":
        return 52
    if freq == "D":
        return 7
    return None


def _future_index(last_date, periods, freq):
    return pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=periods,
        freq=freq,
    )


def _coerce_horizon_to_periods(horizon: int, horizon_unit: str, freq: str) -> int:
    """
    Convert horizon + unit into a number of forecast periods for the chosen frequency.

    Examples:
      freq="MS" & unit="months" -> periods = horizon
      freq="W"  & unit="months" -> periods ~= horizon * 4 (approx)
      freq="D"  & unit="weeks"  -> periods = horizon * 7
    """
    if horizon is None:
        raise ValueError("horizon is required")
    try:
        h = int(horizon)
    except Exception:
        raise ValueError("horizon must be an integer")
    if h <= 0:
        raise ValueError("horizon must be > 0")

    unit = (horizon_unit or "").strip().lower()
    if unit == "":
        # Backward compatibility: treat as periods in chosen freq
        return h

    # Normalize common user inputs
    if unit in ("period", "periods"):
        return h
    if unit in ("month", "months", "mo"):
        if freq == "MS":
            return h
        if freq == "W":
            return max(1, int(round(h * 4)))     # approx 4 weeks / month
        if freq == "D":
            return max(1, int(round(h * 30)))    # approx 30 days / month
        return h
    if unit in ("week", "weeks", "wk", "wks"):
        if freq == "W":
            return h
        if freq == "D":
            return h * 7
        if freq == "MS":
            return max(1, int(round(h / 4)))     # approx
        return h
    if unit in ("day", "days"):
        if freq == "D":
            return h
        if freq == "W":
            return max(1, int(round(h / 7)))
        if freq == "MS":
            return max(1, int(round(h / 30)))
        return h

    # Unknown units: treat as periods
    return h


def _align_and_impute(ts: pd.Series, freq: str, impute_method: str = "linear") -> tuple[pd.Series, dict]:
    """
    Align to a continuous frequency index and impute missing timestamps.

    - Builds full date range from min..max at `freq`
    - Reindexes series
    - Imputes gaps (default linear), then ffill/bfill as safety net
    - Returns (aligned_series, quality_report)

    This is where we become "enterprise-grade" for real-world time series.
    """
    ts = _ensure_datetime_index(ts)
    ts = ts.copy()

    start = ts.index.min()
    end = ts.index.max()

    full_idx = pd.date_range(start=start, end=end, freq=freq)
    aligned = ts.reindex(full_idx)

    missing_mask = aligned.isna()
    missing_points = int(missing_mask.sum())
    total_points = int(len(aligned))
    missing_ratio = float(missing_points / total_points) if total_points > 0 else 0.0

    if missing_points > 0:
        if impute_method == "linear":
            aligned = aligned.interpolate(method="time", limit_direction="both")
        elif impute_method in ("ffill", "pad"):
            aligned = aligned.ffill()
        elif impute_method in ("bfill", "backfill"):
            aligned = aligned.bfill()
        else:
            # Unknown method -> default linear
            aligned = aligned.interpolate(method="time", limit_direction="both")

        # Safety net if still missing
        aligned = aligned.ffill().bfill()

    # After imputation, if still NaN, we can't proceed.
    if aligned.isna().any():
        # Instead of throwing "must be continuous", we throw a precise data-quality error.
        raise ValueError("Series could not be imputed to a fully continuous series for the selected frequency.")

    quality = {
        "aligned_frequency": freq,
        "missing_points": missing_points,
        "total_points": total_points,
        "missing_ratio": missing_ratio,
        "imputation_method": impute_method if missing_points > 0 else None,
    }
    return aligned.astype(float), quality


# ============================================================
# Model Eligibility
# ============================================================

def _allow_prophet(ts: pd.Series, freq: str) -> bool:
    if not PROPHET_AVAILABLE:
        return False
    if freq not in ("MS", "W"):
        return False
    if len(ts) < 24:
        return False
    return True


# ============================================================
# Models
# ============================================================

def _ets_model(ts, freq, periods):
    try:
        season_len = _season_length(freq)
        if season_len and len(ts) < season_len * 2:
            season_len = None

        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add" if season_len else None,
            seasonal_periods=season_len,
        ).fit()

        future_idx = _future_index(ts.index[-1], periods, freq)
        forecast = pd.Series(model.forecast(periods).values, index=future_idx)

        resid = ts - model.fittedvalues
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        return {
            "forecast": forecast,
            "ci_low": forecast - 1.96 * rmse,
            "ci_high": forecast + 1.96 * rmse,
            "info": {"model": "ETS"},
            "metrics": {"rmse": _sanitize_numeric(rmse)},
        }
    except Exception:
        return None


def _linear_model(ts, freq, periods):
    try:
        y = ts.values
        x = np.arange(len(y))

        coef = np.polyfit(x, y, 1)
        trend = coef[0] * x + coef[1]
        resid = y - trend
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        future_x = np.arange(len(y), len(y) + periods)
        future_idx = _future_index(ts.index[-1], periods, freq)
        forecast = pd.Series(coef[0] * future_x + coef[1], index=future_idx)

        return {
            "forecast": forecast,
            "ci_low": forecast - 1.96 * rmse,
            "ci_high": forecast + 1.96 * rmse,
            "info": {"model": "Linear"},
            "metrics": {"rmse": _sanitize_numeric(rmse)},
        }
    except Exception:
        return None


def _prophet_model(ts, freq, periods, cfg):
    if not PROPHET_AVAILABLE:
        return None
    try:
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
    except Exception:
        return None


def _naive(ts, freq, periods):
    last = float(ts.iloc[-1])
    future_idx = _future_index(ts.index[-1], periods, freq)
    forecast = pd.Series([last] * periods, index=future_idx)

    return {
        "forecast": forecast,
        "ci_low": forecast,
        "ci_high": forecast,
        "info": {"model": "Naive"},
        "metrics": {"rmse": None},
    }


# ============================================================
# Auto Model Selection (ENTERPRISE-GRADE)
# ============================================================

def _select_model(results, ts, freq, periods):
    rmses = {
        k: v["metrics"]["rmse"]
        for k, v in results.items()
        if v.get("metrics") and v["metrics"].get("rmse") is not None
    }

    if not rmses:
        return list(results.keys())[0], "No metrics available"

    best = min(rmses.values())
    tolerance = 0.02

    # Guard against divide-by-zero
    if best is None or not math.isfinite(best) or best <= 0:
        winner = min(rmses, key=rmses.get)
        return winner, "RMSE-dominant"

    candidates = [
        k for k, r in rmses.items()
        if r is not None and math.isfinite(r) and (abs(r - best) / best) <= tolerance
    ]

    # Monthly multi-step: prefer ETS if close to best
    if "ETS" in candidates and freq == "MS" and periods > 1:
        return "ETS", "Structural preference: season-capable model"

    if "Prophet" in candidates and _allow_prophet(ts, freq):
        return "Prophet", "Nonlinear model preferred"

    winner = min(rmses, key=rmses.get)
    return winner, "RMSE-dominant"


# ============================================================
# Main Engine Entry (LOCKED)
# ============================================================

def forecast_series(
    ts: pd.Series,
    freq_choice: str | None,
    horizon: int,
    horizon_unit: str,
    prophet_cfg: dict,
    model_strategy: str,
):
    start = time.perf_counter()

    # 1) Sanitize (but do NOT require continuity here)
    ts = _safe_series(ts)

    # 2) Normalize frequency
    freq = normalize_frequency(freq_choice, ts.index.freqstr)

    # 3) Convert horizon+unit into forecast periods
    periods = _coerce_horizon_to_periods(horizon, horizon_unit, freq)

    # 4) Align + impute to continuous series (enterprise-grade)
    #    Default policy: linear interpolation + ffill/bfill safety net.
    ts_aligned, quality = _align_and_impute(ts, freq=freq, impute_method="linear")

    comparison = {}

    ets = _ets_model(ts_aligned, freq, periods)
    if ets:
        comparison["ETS"] = ets

    lin = _linear_model(ts_aligned, freq, periods)
    if lin:
        comparison["Linear"] = lin

    if model_strategy.lower() in ("auto", "prophet") and _allow_prophet(ts_aligned, freq):
        prop = _prophet_model(ts_aligned, freq, periods, prophet_cfg)
        if prop:
            comparison["Prophet"] = prop

    if not comparison:
        comparison["Naive"] = _naive(ts_aligned, freq, periods)

    winner, rationale = _select_model(comparison, ts_aligned, freq, periods)
    selected = comparison[winner]

    # Risk escalation: gaps mean elevated risk, even if we imputed.
    missing_ratio = float(quality.get("missing_ratio", 0.0) or 0.0)
    risk_flag = "Normal"
    if winner == "Naive":
        risk_flag = "Fallback"
    elif missing_ratio >= 0.20:
        risk_flag = "High"
    elif missing_ratio >= 0.05:
        risk_flag = "Elevated"

    audit = {
        "requested_model": model_strategy,
        "selected_model": winner,
        "selection_rationale": rationale,
        "execution_time_ms": int((time.perf_counter() - start) * 1000),
        "risk_flag": risk_flag,
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
