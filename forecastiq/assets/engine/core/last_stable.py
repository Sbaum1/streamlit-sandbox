import numpy as np
import pandas as pd
from datetime import timedelta
import time
import signal
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
# Prophet Timeout Guard (cross-platform)
# ---------------------------

class ProphetTimeout(Exception):
    pass


@contextmanager
def prophet_time_limit(seconds: int):
    """
    Cross-platform timeout guard:
    - On Unix, uses SIGALRM.
    - On Windows (no SIGALRM), this becomes a no-op; use thread timeout in _prophet_model_full.
    """
    if hasattr(signal, "SIGALRM"):
        def signal_handler(signum, frame):
            raise ProphetTimeout("Prophet execution timed out")

        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: no SIGALRM; handled via ThreadPoolExecutor timeout
        yield


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
        return pd.date_range(
            start=last_date,
            periods=horizon + 1,
            freq=freq
        )[1:]

    return pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq="D",
    )


def _set_index_freq_if_possible(ts: pd.Series, freq: str) -> pd.Series:
    """
    Avoids statsmodels 'No frequency information' warnings when possible,
    without changing the data or resampling.
    """
    try:
        ts2 = ts.copy()
        ts2.index = pd.DatetimeIndex(ts2.index)

        if freq:
            try:
                ts2.index.freq = pd.tseries.frequencies.to_offset(freq)
            except Exception:
                # If pandas can't set it safely, leave it alone.
                pass

        return ts2
    except Exception:
        return ts


def _sanitize_series_for_json(s: pd.Series | None) -> pd.Series | None:
    """
    Starlette/FastAPI JSON encoding rejects NaN/Inf when allow_nan=False.
    Convert any non-finite values to None (object dtype) so to_dict() is JSON-safe.
    """
    if s is None:
        return None

    try:
        vals = s.to_numpy()
        out = []
        for v in vals:
            try:
                fv = float(v)
            except Exception:
                out.append(None)
                continue
            if np.isfinite(fv):
                out.append(fv)
            else:
                out.append(None)
        return pd.Series(out, index=s.index, dtype="object")
    except Exception:
        # Last-resort: if anything goes sideways, drop to None series
        return pd.Series([None] * len(s), index=s.index, dtype="object")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------
# Model Eligibility Rules
# ---------------------------

def _should_use_prophet(ts: pd.Series, freq: str) -> bool:
    if not PROPHET_AVAILABLE:
        return False

    if freq not in ("M", "MS", "W", "W-SUN"):
        return False

    if len(ts) < 12:
        return False

    resid_std = ts.diff().dropna().std()
    level_std = ts.std()

    if level_std == 0 or pd.isna(level_std):
        return False

    return (resid_std / level_std) > 0.15


# ---------------------------
# ETS (PRIMARY MODEL)
# ---------------------------

def _ets_model_full(ts, freq, horizon, horizon_unit):
    if len(ts) < 6:
        return None

    try:
        ts = _set_index_freq_if_possible(ts, freq)

        season_len = _get_season_length(freq)
        if season_len and len(ts) < season_len * 2:
            season_len = None

        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add" if season_len else None,
            seasonal_periods=season_len,
        ).fit()

        future_idx = _build_future_index(
            ts.index[-1],
            horizon,
            freq,
            horizon_unit,
        )

        fc = model.forecast(horizon)
        resid = ts - model.fittedvalues
        resid_std = float(resid.std()) if resid is not None else float("nan")

        # Ensure all series are aligned and JSON-safe downstream
        fc_s = pd.Series(np.asarray(fc, dtype=float), index=future_idx)
        ci_low = pd.Series(np.asarray(fc_s - 1.96 * resid_std, dtype=float), index=future_idx)
        ci_high = pd.Series(np.asarray(fc_s + 1.96 * resid_std, dtype=float), index=future_idx)

        return {
            "forecast": fc_s,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "info": {"model": "ETS"},
            "metrics": {"rmse": resid_std},
        }
    except Exception:
        return None


# ---------------------------
# Linear (SECONDARY FALLBACK)
# ---------------------------

def _linear_model_full(ts, horizon, horizon_unit):
    if len(ts) < 4:
        return None

    try:
        y = ts.values.astype(float)
        x = np.arange(len(y), dtype=float)

        coef = np.polyfit(x, y, 1)
        future_x = np.arange(len(y), len(y) + horizon, dtype=float)
        fc = coef[0] * future_x + coef[1]

        freq = ts.index.freqstr or infer_frequency(ts.index.to_series())
        future_idx = _build_future_index(
            ts.index[-1],
            horizon,
            freq,
            horizon_unit,
        )

        resid = y - (coef[0] * x + coef[1])
        resid_std = float(np.std(resid))

        fc_s = pd.Series(np.asarray(fc, dtype=float), index=future_idx)
        ci_low = pd.Series(np.asarray(fc_s - 1.96 * resid_std, dtype=float), index=future_idx)
        ci_high = pd.Series(np.asarray(fc_s + 1.96 * resid_std, dtype=float), index=future_idx)

        return {
            "forecast": fc_s,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "info": {"model": "Linear"},
            "metrics": {"rmse": resid_std},
        }
    except Exception:
        return None


# ---------------------------
# Prophet (SAFE + CONFIGURED)
# ---------------------------

def _prophet_fit_predict(ts, freq, horizon, safe_cfg):
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


def _prophet_model_full(ts, freq, horizon, prophet_cfg: dict, audit: dict):
    if not PROPHET_AVAILABLE or len(ts) < 8:
        return None

    # Whitelisted config
    allowed_keys = {
        "growth",
        "seasonality_mode",
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "weekly_seasonality",
        "yearly_seasonality",
        "daily_seasonality",
    }

    safe_cfg = {k: v for k, v in (prophet_cfg or {}).items() if k in allowed_keys}
    audit["prophet_config_used"] = safe_cfg

    try:
        ts = _set_index_freq_if_possible(ts, freq)

        # Unix: SIGALRM guard; Windows: thread timeout
        if hasattr(signal, "SIGALRM"):
            with prophet_time_limit(2):
                return _prophet_fit_predict(ts, freq, horizon, safe_cfg)

        # Windows / no SIGALRM
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_prophet_fit_predict, ts, freq, horizon, safe_cfg)
            try:
                return fut.result(timeout=2)
            except FuturesTimeoutError:
                audit["prophet_timeout"] = True
                return None

    except ProphetTimeout:
        audit["prophet_timeout"] = True
        return None
    except Exception:
        return None


# ---------------------------
# Naive (LAST RESORT)
# ---------------------------

def _naive_forecast(ts, horizon, horizon_unit):
    last = float(ts.iloc[-1])
    freq = ts.index.freqstr or infer_frequency(ts.index.to_series())
    future_idx = _build_future_index(ts.index[-1], horizon, freq, horizon_unit)

    return {
        "forecast": pd.Series([last] * horizon, index=future_idx),
        "ci_low": None,
        "ci_high": None,
        "info": {"model": "Naive"},
        "metrics": None,
    }


# ---------------------------
# Lightweight Backtest Scoring (for Auto selection)
# ---------------------------

def _holdout_size(ts: pd.Series, horizon: int) -> int:
    n = len(ts)
    if n < 8:
        return 0
    # Small but meaningful holdout; keeps performance fast.
    candidate = min(max(2, min(6, horizon)), max(2, n // 4))
    if candidate >= n:
        candidate = max(1, n - 3)
    return int(candidate)


def _score_candidate_rmse(name: str, ts: pd.Series, freq: str, horizon_unit: str, prophet_cfg: dict) -> float | None:
    k = _holdout_size(ts, horizon=6)  # scoring horizon capped internally
    if k <= 0 or len(ts) <= k + 2:
        return None

    train = ts.iloc[:-k]
    test = ts.iloc[-k:]
    y_true = test.values.astype(float)

    if name == "ETS":
        out = _ets_model_full(train, freq, k, "Periods")
        if not out or out.get("forecast") is None:
            return None
        y_pred = np.asarray(out["forecast"].values, dtype=float)[:k]
        return _rmse(y_true[:len(y_pred)], y_pred)

    if name == "Prophet":
        out_audit = {}
        out = _prophet_model_full(train, freq, k, prophet_cfg, out_audit)
        if not out or out.get("forecast") is None:
            return None
        y_pred = np.asarray(out["forecast"].values, dtype=float)[:k]
        return _rmse(y_true[:len(y_pred)], y_pred)

    if name == "Linear":
        out = _linear_model_full(train, k, "Periods")
        if not out or out.get("forecast") is None:
            return None
        y_pred = np.asarray(out["forecast"].values, dtype=float)[:k]
        return _rmse(y_true[:len(y_pred)], y_pred)

    return None


# ---------------------------
# Main Entry Point (STABLE CONTRACT)
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

    audit = {
        "requested_model": model_strategy,
        "selected_model": None,
        "reason": None,
        "fallback_chain": [],
        "execution_time_ms": None,
        # Additive diagnostics (safe for UI use)
        "model_scores": {},
        "selection_rule": None,
        # Prophet audit keys remain additive
        "prophet_config_used": {},
    }

    # Frequency
    freq = ts.index.freqstr or infer_frequency(ts.index.to_series())
    if freq_choice == "Monthly":
        freq = freq or "MS"
    elif freq_choice == "Weekly":
        freq = freq or "W"

    comparison = {}

    # ---------------------------
    # Explicit model requests
    # ---------------------------
    if model_strategy == "Prophet":
        audit["fallback_chain"].append("Prophet")
        prop = _prophet_model_full(ts, freq, horizon, prophet_cfg, audit)
        if prop:
            audit["selected_model"] = "Prophet"
            audit["reason"] = "Explicit request"
            audit["selection_rule"] = "explicit_prophet"
            comparison["Prophet"] = prop
            audit["execution_time_ms"] = int((time.perf_counter() - start_time) * 1000)

            # JSON-safe outputs
            return (
                _sanitize_series_for_json(prop["forecast"]),
                _sanitize_series_for_json(prop["ci_low"]),
                _sanitize_series_for_json(prop["ci_high"]),
                prop["info"],
                prop["metrics"],
                {k: v for k, v in comparison.items()},
                audit,
            )

    # ---------------------------
    # Candidates (Auto)
    # ---------------------------
    prop = None
    if model_strategy == "Auto" and _should_use_prophet(ts, freq):
        audit["fallback_chain"].append("Prophet")
        prop = _prophet_model_full(ts, freq, horizon, prophet_cfg, audit)
        if prop:
            comparison["Prophet"] = prop

    audit["fallback_chain"].append("ETS")
    ets = _ets_model_full(ts, freq, horizon, horizon_unit)
    if ets:
        comparison["ETS"] = ets

    # Optional: linear as an extra fallback candidate (only if needed)
    lin = None
    if not ets:
        audit["fallback_chain"].append("Linear")
        lin = _linear_model_full(ts, horizon, horizon_unit)
        if lin:
            comparison["Linear"] = lin

    # ---------------------------
    # Selection: Prophet wins if it beats ETS by holdout RMSE
    # ---------------------------
    selected_key = None

    if model_strategy == "Auto" and ("ETS" in comparison or "Prophet" in comparison):
        # Score available candidates (fast holdout)
        if "ETS" in comparison:
            audit["model_scores"]["ETS"] = _score_candidate_rmse("ETS", ts, freq, horizon_unit, prophet_cfg)
        if "Prophet" in comparison:
            audit["model_scores"]["Prophet"] = _score_candidate_rmse("Prophet", ts, freq, horizon_unit, prophet_cfg)

        ets_score = audit["model_scores"].get("ETS")
        prop_score = audit["model_scores"].get("Prophet")

        if prop_score is not None and ets_score is not None:
            # Prophet wins only if strictly better (lower RMSE)
            if prop_score < ets_score:
                selected_key = "Prophet"
                audit["reason"] = "Prophet beats ETS on holdout RMSE"
                audit["selection_rule"] = "prophet_over_ets_rmse"
            else:
                selected_key = "ETS"
                audit["reason"] = "ETS retained (best holdout RMSE)"
                audit["selection_rule"] = "ets_best_rmse"
        elif "ETS" in comparison:
            selected_key = "ETS"
            audit["reason"] = "Primary model"
            audit["selection_rule"] = "ets_default"
        else:
            selected_key = "Prophet"
            audit["reason"] = "Only viable model"
            audit["selection_rule"] = "prophet_only"

    # ---------------------------
    # Final fallback chain
    # ---------------------------
    if selected_key is None:
        if "ETS" in comparison:
            selected_key = "ETS"
            audit["reason"] = "Primary model"
            audit["selection_rule"] = "ets_default"
        elif "Prophet" in comparison:
            selected_key = "Prophet"
            audit["reason"] = "Only viable model"
            audit["selection_rule"] = "prophet_only"
        elif "Linear" in comparison:
            selected_key = "Linear"
            audit["reason"] = "ETS unavailable"
            audit["selection_rule"] = "linear_fallback"
        else:
            audit["fallback_chain"].append("Naive")
            naive = _naive_forecast(ts, horizon, horizon_unit)
            audit["selected_model"] = "Naive"
            audit["reason"] = "All models unavailable"
            audit["selection_rule"] = "naive_last_resort"
            audit["execution_time_ms"] = int((time.perf_counter() - start_time) * 1000)

            return (
                _sanitize_series_for_json(naive["forecast"]),
                _sanitize_series_for_json(naive["ci_low"]),
                _sanitize_series_for_json(naive["ci_high"]),
                naive["info"],
                naive["metrics"],
                {k: v for k, v in comparison.items()},
                audit,
            )

    selected = comparison[selected_key]
    audit["selected_model"] = selected["info"]["model"]
    audit["execution_time_ms"] = int((time.perf_counter() - start_time) * 1000)

    # JSON-safe return (STABLE: 7 values)
    return (
        _sanitize_series_for_json(selected["forecast"]),
        _sanitize_series_for_json(selected["ci_low"]),
        _sanitize_series_for_json(selected["ci_high"]),
        selected["info"],
        selected["metrics"],
        {k: v for k, v in comparison.items()},
        audit,
    )
