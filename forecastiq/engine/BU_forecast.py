# ==================================================
# FILE: forecastiq/engine/forecast.py
# VERSION: 2.0.0
# ROLE: SENTINEL ENGINE ADAPTER
# UPDATED: Phase 4 — Replaces 5-model engine with
#          sentinel_engine.run_all_models() call.
#          Preserves existing UI session state contract.
# ==================================================
#
# ADAPTER CONTRACT:
#   Input : committed_df (date, value), frequency, horizons, tier
#   Output: all existing session state keys preserved +
#           new sentinel_* keys populated
#
# OUTPUT MAPPING (legacy keys → sentinel outputs):
#   latest_metrics      ← per-model MASE/RMSE/status DataFrame
#   latest_forecasts    ← {model_name: pd.Series} future forecasts
#   latest_intervals    ← {model_name: (lower, upper) or None}
#   latest_model_name   ← "Primary Ensemble"
#   latest_forecast_df  ← Primary Ensemble forecast Series
#
# FALLBACK:
#   If sentinel_engine fails for any reason, raises with
#   a clear message. No silent fallback to old engine.
# ==================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure sentinel_engine is importable from parent directory ────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sentinel_engine import run_all_models
from sentinel_engine.sentinel_config import set_tier as sentinel_set_tier


# ── Frequency mapping: ForecastIQ → sentinel_engine ──────────────────────────
_FREQ_MAP = {
    "Monthly": "MS",
    "Weekly":  "W",
    "Daily":   "D",
}


def run_forecast(
    df:               pd.DataFrame,
    frequency:        str,
    backtest_horizon: int,
    forecast_horizon: int,
    active_tier:      str = "Enterprise",
    confidence_level: float = 0.95,
    exog_df:          "pd.DataFrame | None" = None,
) -> dict:
    """
    Run sentinel_engine and return a results bundle compatible
    with the ForecastIQ session state contract.

    Returns dict with keys:
        legacy          — dict matching old run_all_models() output shape
        sentinel        — raw sentinel_engine results dict
        engine_meta     — _engine metadata block
        cert_metadata   — per-model cert summary list
    """
    # ── Configure sentinel tier ───────────────────────────────────────────────
    sentinel_set_tier(active_tier)

    # ── Run sentinel engine ───────────────────────────────────────────────────
    results = run_all_models(
        df               = df,
        horizon          = forecast_horizon,
        confidence_level = confidence_level,
    )

    # ── Extract engine metadata ───────────────────────────────────────────────
    engine_meta = results.get("_engine", {})

    # ── Build legacy metrics DataFrame ───────────────────────────────────────
    metrics_rows = []
    forecasts    = {}
    intervals    = {}

    skip = {"_engine", "_failures", "Primary Ensemble", "Stacked Ensemble"}

    for name, result in results.items():
        if name in skip or name.startswith("_"):
            continue
        if not isinstance(result, dict):
            continue

        status = result.get("status", "failed")
        diag   = result.get("diagnostic_only", False)

        if diag:
            continue

        m = result.get("metrics", {}) or {}

        mase = m.get("MASE")
        rmse = m.get("RMSE")
        mae  = m.get("MAE")

        metrics_rows.append({
            "model":  name,
            "MASE":   mase,
            "MAE":    mae  if mae  is not None else np.nan,
            "RMSE":   rmse if rmse is not None else np.nan,
            "MAPE":   m.get("MAPE", np.nan),
            "status": "OK" if status == "success" else "FAILED",
            "tier":   result.get("metadata", {}).get("active_tier", active_tier),
            "notes":  result.get("metadata", {}).get("stacker_version", ""),
        })

        # ── Forecast series (future periods only) ─────────────────────────
        fdf = result.get("forecast_df")
        if fdf is not None and not fdf.empty:
            future = fdf[fdf["actual"].isna()].copy()
            if not future.empty:
                s = pd.Series(
                    future["forecast"].values,
                    index=pd.to_datetime(future["date"].values),
                    name=name,
                )
                forecasts[name] = s

                # CI
                if "ci_low" in future.columns and "ci_high" in future.columns:
                    lower = pd.Series(future["ci_low"].values,  index=s.index)
                    upper = pd.Series(future["ci_high"].values, index=s.index)
                    if lower.notna().any() and upper.notna().any():
                        intervals[name] = (lower, upper)
                    else:
                        intervals[name] = None
                else:
                    intervals[name] = None

    # ── Primary Ensemble ──────────────────────────────────────────────────────
    pe = results.get("Primary Ensemble", {})
    pe_series = None
    pe_lower  = None
    pe_upper  = None

    if pe and pe.get("status") == "success":
        pe_fdf = pe.get("forecast_df")
        if pe_fdf is not None and not pe_fdf.empty:
            pe_future = pe_fdf[pe_fdf["actual"].isna()].copy()
            if not pe_future.empty:
                pe_series = pd.Series(
                    pe_future["forecast"].values,
                    index=pd.to_datetime(pe_future["date"].values),
                    name="Primary Ensemble",
                )
                forecasts["Primary Ensemble"] = pe_series

                if "ci_low" in pe_future.columns:
                    pe_lower = pd.Series(pe_future["ci_low"].values,  index=pe_series.index)
                    pe_upper = pd.Series(pe_future["ci_high"].values, index=pe_series.index)
                    intervals["Primary Ensemble"] = (pe_lower, pe_upper)

        pe_m = pe.get("metrics", {}) or {}
        metrics_rows.append({
            "model":  "Primary Ensemble",
            "MASE":   pe_m.get("MASE"),
            "MAE":    pe_m.get("MAE",  np.nan),
            "RMSE":   pe_m.get("RMSE", np.nan),
            "MAPE":   pe_m.get("MAPE", np.nan),
            "status": "OK",
            "tier":   "All",
            "notes":  pe.get("metadata", {}).get("aggregation_method", ""),
        })

    # ── Stacked Ensemble ──────────────────────────────────────────────────────
    se = results.get("Stacked Ensemble", {})
    se_series = None

    if se and se.get("status") == "success":
        se_fdf = se.get("forecast_df")
        if se_fdf is not None and not se_fdf.empty:
            se_future = se_fdf[se_fdf["actual"].isna()].copy()
            if not se_future.empty:
                se_series = pd.Series(
                    se_future["forecast"].values,
                    index=pd.to_datetime(se_future["date"].values),
                    name="Stacked Ensemble",
                )
                forecasts["Stacked Ensemble"] = se_series
                if "ci_low" in se_future.columns:
                    intervals["Stacked Ensemble"] = (
                        pd.Series(se_future["ci_low"].values,  index=se_series.index),
                        pd.Series(se_future["ci_high"].values, index=se_series.index),
                    )

    # ── Cert metadata ─────────────────────────────────────────────────────────
    cert_metadata = []
    for name, result in results.items():
        if name.startswith("_") or not isinstance(result, dict):
            continue
        if result.get("status") != "success":
            continue
        m    = result.get("metrics", {}) or {}
        meta = result.get("metadata", {}) or {}
        ea   = result.get("executive_assessment", {}) or {}
        cert_metadata.append({
            "model":        name,
            "MASE":         m.get("MASE"),
            "cert_tier":    ea.get("readiness_tier", "—"),
            "ci_method":    meta.get("ci_method", "—"),
            "active_tier":  meta.get("active_tier", active_tier),
            "stacker":      meta.get("stacker_active", False),
        })

    metrics_df = pd.DataFrame(metrics_rows)

    # ── Default winner = Primary Ensemble ────────────────────────────────────
    winner = "Primary Ensemble"

    # ── Build forecast_viz_df — required by locked forecast_viz.py ───────────
    # Shape: date, actual, forecast, lower, upper
    # Combines historical actuals (future=NaN) + forward forecast periods
    hist_df = df[["date", "value"]].copy()
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df.columns = ["date", "actual"]
    hist_df["forecast"] = np.nan
    hist_df["lower"]    = np.nan
    hist_df["upper"]    = np.nan

    if pe_series is not None:
        fwd_rows = []
        pe_ci = intervals.get("Primary Ensemble")
        for i, (dt, fval) in enumerate(zip(pe_series.index, pe_series.values)):
            lo = pe_ci[0].iloc[i] if pe_ci is not None else np.nan
            hi = pe_ci[1].iloc[i] if pe_ci is not None else np.nan
            fwd_rows.append({
                "date":     dt,
                "actual":   np.nan,
                "forecast": fval,
                "lower":    lo,
                "upper":    hi,
            })
        fwd_df = pd.DataFrame(fwd_rows)
        forecast_viz_df = pd.concat([hist_df, fwd_df], ignore_index=True)
    else:
        forecast_viz_df = hist_df

    return {
        "legacy": {
            "metrics_df":       metrics_df,
            "forecasts":        forecasts,
            "intervals":        intervals,
            "winner":           winner,
            "winner_forecast":  pe_series,
            "forecast_viz_df":  forecast_viz_df,   # ← DataFrame for forecast_viz.py
            "primary_df":       pe_series,
            "stacked_df":       se_series,
        },
        "sentinel":      results,
        "engine_meta":   engine_meta,
        "cert_metadata": cert_metadata,
    }
