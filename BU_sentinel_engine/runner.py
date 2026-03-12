# ==================================================
# FILE: sentinel_engine/runner.py
# VERSION: 2.0.0
# ROLE: MODEL EXECUTION ORCHESTRATOR
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - All logic ported from streamlit_sandbox/analysis/forecast_runner.py
# - Three fixes applied vs original:
#     FIX 1: forecast_completed/forecast_executed key conflict removed
#             (engine has no session state — conflict cannot exist)
#     FIX 2: Stress test operates on copy — never mutates original output
#     FIX 3: diagnostic_only and ensemble_member flags read from registry
#             (X-13 exclusion governed at registry level, not ad-hoc)
# - Input validation is strict — fails fast with clear messages
# - Actuals injection is non-destructive
# - Forward boundary validation prevents lookahead contamination
# - Metric normalization uses authoritative key mapping
# - Readiness tier and confidence posture engines fully preserved
# ==================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .contracts import ForecastResult, ENGINE_VERSION
from .registry  import get_model_registry


# --------------------------------------------------
# BACKTEST MINIMUM (mirrors backtest_engine.py)
# --------------------------------------------------

MIN_OBSERVATIONS = 36


# ==================================================
# FAILURE RECORD
# ==================================================

def _build_failure_record(model_name: str, error: Exception) -> dict:
    return {
        "model_name":    model_name,
        "error_type":    type(error).__name__,
        "error_message": str(error),
    }


# ==================================================
# INPUT VALIDATION
# ==================================================

def _validate_input(df: pd.DataFrame) -> pd.DataFrame:

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("Input dataframe must contain 'date' and 'value'.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    df = df.sort_values("date").reset_index(drop=True)

    inferred = pd.infer_freq(df["date"])
    if inferred not in ("MS", "M"):
        raise ValueError(
            f"Monthly frequency required. Inferred frequency: '{inferred}'."
        )

    df = df.set_index("date").asfreq(inferred).reset_index()

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    return df


# ==================================================
# ACTUALS INJECTION (NON-DESTRUCTIVE)
# ==================================================

def _inject_actuals(
    forecast_df:    pd.DataFrame,
    historical_df:  pd.DataFrame,
) -> pd.DataFrame:

    df = forecast_df.copy()

    if "actual" not in df.columns:
        df["actual"] = pd.NA

    hist = historical_df[["date", "value"]].rename(
        columns={"value": "actual_hist"}
    )

    merged = pd.merge(df, hist, on="date", how="left")
    merged["actual"] = merged["actual"].combine_first(merged["actual_hist"])
    merged = merged.drop(columns=["actual_hist"])

    if merged["date"].duplicated().any():
        raise ValueError("Duplicate dates after actual injection.")

    return merged


# ==================================================
# FORWARD BOUNDARY CHECK
# ==================================================

def _validate_forward_boundary(forecast_df: pd.DataFrame) -> None:

    df     = forecast_df.sort_values("date")
    hist   = df[df["actual"].notna()]
    future = df[df["actual"].isna()]

    if not hist.empty and not future.empty:
        if future["date"].min() <= hist["date"].max():
            raise ValueError(
                "Forecast horizon overlaps historical period — "
                "lookahead contamination detected."
            )


# ==================================================
# METRIC NORMALIZATION (AUTHORITATIVE CONTRACT)
# ==================================================

def _normalize_metric_keys(metrics: dict) -> dict:

    if not isinstance(metrics, dict):
        return {}

    mapping = {
        "mae":          "MAE",
        "rmse":         "RMSE",
        "mape":         "MAPE",
        "bias":         "Bias",
        "mase":         "MASE",
        "theils_u":     "Theils_U",
        "ci_coverage":  "CI_Coverage",
        "smape":        "SMAPE",
        "folds":        "Folds",
        "observations": "Observations",
        "mean_level":   "Mean_Level",
    }

    return {mapping.get(k, k): v for k, v in metrics.items()}


# ==================================================
# READINESS TIER ENGINE
# ==================================================

def _assign_readiness(metrics: dict, confidence_level: float) -> str:

    if metrics.get("eligible") is False:
        return "Ineligible — Minimum Data Not Met"

    mase       = metrics.get("MASE")
    theils_u   = metrics.get("Theils_U")
    coverage   = metrics.get("CI_Coverage")
    bias       = metrics.get("Bias")
    folds      = metrics.get("Folds")
    mean_level = metrics.get("Mean_Level")

    if mase is None or theils_u is None:
        return "Unscored"

    if coverage is not None and coverage < 0.50:
        return "Tier 4 — Structural Failure"

    bias_ok = True
    if bias is not None and mean_level not in (None, 0):
        bias_ratio = abs(bias) / abs(mean_level)
        if bias_ratio > 0.02:
            bias_ok = False

    if folds is not None and folds < 3:
        return "Tier 3 — Weak (Limited Fold Validation)"

    if (
        mase < 0.8
        and theils_u < 1.0
        and coverage is not None
        and abs(coverage - confidence_level) <= 0.05
        and bias_ok
    ):
        return "Tier 1 — Production Ready"

    if mase < 1.0:
        return "Tier 2 — Acceptable"

    return "Tier 3 — Weak"


# ==================================================
# EXECUTIVE CONFIDENCE ENGINE
# ==================================================

def _assign_confidence(metrics: dict, confidence_level: float) -> dict:

    if metrics.get("eligible") is False:
        return {
            "confidence_posture": "Not Eligible",
            "risk_flags":         ["Minimum data threshold not met"],
            "decision_guidance":  "Increase historical data before executive use.",
        }

    mase       = metrics.get("MASE")
    theils_u   = metrics.get("Theils_U")
    coverage   = metrics.get("CI_Coverage")
    bias       = metrics.get("Bias")
    mean_level = metrics.get("Mean_Level")

    risk_flags: List[str] = []

    if mase is not None and mase > 1.0:
        risk_flags.append("Error exceeds naïve baseline")

    if coverage is not None:
        deviation = abs(coverage - confidence_level)
        if deviation > 0.10:
            risk_flags.append("Confidence interval materially miscalibrated")
        elif deviation > 0.05:
            risk_flags.append("Minor confidence interval drift")
        if coverage < 0.50:
            risk_flags.append("Confidence band structural failure")

    if bias is not None and mean_level not in (None, 0):
        bias_ratio = abs(bias) / abs(mean_level)
        if bias_ratio > 0.02:
            risk_flags.append("Structural forecast bias (>2%)")

    if mase is not None and theils_u is not None:
        if mase < 0.8 and theils_u < 1.0 and not risk_flags:
            posture  = "High Confidence — Production Safe"
            guidance = "Model suitable for executive planning use."
        elif mase < 1.0:
            posture  = "Moderate Confidence — Monitor"
            guidance = "Model acceptable but monitor volatility and calibration."
        else:
            posture  = "Low Confidence — Elevated Risk"
            guidance = "Exercise caution in executive decisions."
    else:
        posture  = "Unscored"
        guidance = "Metrics incomplete."

    return {
        "confidence_posture": posture,
        "risk_flags":         risk_flags,
        "decision_guidance":  guidance,
    }


# ==================================================
# STRESS TEST (FIX 2 — OPERATES ON COPY)
# ==================================================

def apply_stress(
    results:      Dict[str, Any],
    stress_pct:   float = 0.15,
) -> Dict[str, Any]:
    """
    Apply stress widening to confidence intervals.

    FIX 2 vs original: operates on a deep copy of results —
    original forecast output in session state is never mutated.

    Args:
        results    : Output dict from run_all_models()
        stress_pct : Fractional CI widening (default 0.15 = +15%)

    Returns:
        New results dict with stressed CIs — original is untouched.
    """

    stressed = {}

    for name, result in results.items():

        if name.startswith("_") or not isinstance(result, dict):
            stressed[name] = result
            continue

        if result.get("status") != "success":
            stressed[name] = result
            continue

        df_out = result.get("forecast_df")

        if df_out is None or df_out.empty:
            stressed[name] = result
            continue

        # FIX 2: copy before mutating
        df_stressed          = df_out.copy()
        width                = df_stressed["ci_high"] - df_stressed["ci_low"]
        shock                = width * stress_pct
        df_stressed["ci_low"]  = df_stressed["ci_low"]  - shock
        df_stressed["ci_high"] = df_stressed["ci_high"] + shock

        stressed[name] = {**result, "forecast_df": df_stressed}

    return stressed


# ==================================================
# BACKTEST STUB
# (Replace with import from sentinel_engine.backtest
#  once backtest_engine.py is ported in Phase 3)
# ==================================================

def _run_backtest_stub(
    df:               pd.DataFrame,
    model_runner:     Any,
    horizon:          int,
    confidence_level: float,
) -> dict:
    """
    Placeholder — returns empty metrics dict.
    Will be replaced by full backtest engine in Phase 3.
    """
    return {}


# ==================================================
# DIAGNOSTICS STUB
# (Replace with import from sentinel_engine.diagnostics
#  once diagnostics.py is ported in Phase 3)
# ==================================================

def _compute_diagnostics_stub(
    forecast_df: pd.DataFrame,
    metrics:     dict,
) -> dict:
    """
    Placeholder — returns empty diagnostics dict.
    Will be replaced by full diagnostics engine in Phase 3.
    """
    return {}


# ==================================================
# MAIN RUNNER
# ==================================================

def run_all_models(
    df:               pd.DataFrame,
    horizon:          int,
    confidence_level: float,
    backtest_fn:      Optional[Any] = None,
    diagnostics_fn:   Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute all registered models and return scored results.

    Governance:
    - Validates input strictly before any model runs
    - Respects diagnostic_only flag from registry (FIX 3)
    - Injects actuals non-destructively
    - Validates forward boundary on every forecast
    - Normalizes all metric keys to authoritative contract
    - Assigns readiness tier and executive confidence posture
    - Collects failures without halting execution
    - Never mutates input DataFrame

    Args:
        df               : Historical DataFrame ('date', 'value')
        horizon          : Forecast periods ahead
        confidence_level : Prediction interval confidence (e.g. 0.9)
        backtest_fn      : Optional backtest callable — defaults to stub
                           (inject full backtest_engine in Phase 3)
        diagnostics_fn   : Optional diagnostics callable — defaults to stub
                           (inject full diagnostics engine in Phase 3)

    Returns:
        Dict of model_name -> result dict.
        Failed models return status='failed' with error details.
        '_failures' key contains list of all failure records.
        '_engine' key contains engine metadata.
    """

    # ── Inject stubs if not provided ────────────────────────────────────────
    if backtest_fn   is None:
        backtest_fn   = _run_backtest_stub
    if diagnostics_fn is None:
        diagnostics_fn = _compute_diagnostics_stub

    results:  Dict[str, Any] = {}
    failures: List[dict]     = []

    # ── Validate input ───────────────────────────────────────────────────────
    hist_df           = _validate_input(df)
    observation_count = len(hist_df)

    # ── Execute all registered models ────────────────────────────────────────
    for model_meta in get_model_registry():

        name            = model_meta["name"]
        runner          = model_meta["runner"]
        diagnostic_only = model_meta.get("diagnostic_only", False)  # FIX 3

        try:

            output = runner(
                df               = hist_df,
                horizon          = horizon,
                confidence_level = confidence_level,
            )

            if not isinstance(output, ForecastResult):
                raise TypeError(
                    f"{name} did not return ForecastResult. "
                    f"Got: {type(output).__name__}"
                )

            forecast_df         = output.forecast_df.copy()
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            # ── Diagnostic-only path (FIX 3) ────────────────────────────────
            if diagnostic_only:
                results[name] = {
                    "status":               "success",
                    "forecast_df":          forecast_df,
                    "metrics":              {},
                    "diagnostics":          {},
                    "executive_assessment": {},
                    "metadata":             output.metadata or {},
                    "diagnostic_only":      True,
                }
                continue

            # ── Production path ──────────────────────────────────────────────
            forecast_df = _inject_actuals(forecast_df, hist_df)
            _validate_forward_boundary(forecast_df)

            # ── Backtest / metrics ───────────────────────────────────────────
            if observation_count < MIN_OBSERVATIONS:
                raw_metrics = {
                    "eligible":     False,
                    "reason":       f"Minimum {MIN_OBSERVATIONS} observations required.",
                    "observations": observation_count,
                }
            else:
                raw_metrics = backtest_fn(
                    df               = hist_df,
                    model_runner     = runner,
                    horizon          = horizon,
                    confidence_level = confidence_level,
                )

                if not isinstance(raw_metrics, dict):
                    raise ValueError(
                        f"Backtest engine returned {type(raw_metrics).__name__}, "
                        "expected dict."
                    )

                raw_metrics["eligible"]     = True
                raw_metrics["observations"] = observation_count

            normalized_metrics = _normalize_metric_keys(raw_metrics)

            readiness         = _assign_readiness(normalized_metrics, confidence_level)
            confidence_bundle = _assign_confidence(normalized_metrics, confidence_level)
            diagnostics       = diagnostics_fn(
                forecast_df = forecast_df,
                metrics     = normalized_metrics,
            )

            metadata                     = dict(output.metadata or {})
            metadata["confidence_level"] = confidence_level
            metadata["engine_version"]   = ENGINE_VERSION

            results[name] = {
                "status":      "success",
                "forecast_df": forecast_df,
                "metrics":     normalized_metrics,
                "diagnostics": diagnostics,
                "executive_assessment": {
                    "readiness_tier": readiness,
                    **confidence_bundle,
                },
                "metadata":         metadata,
                "diagnostic_only":  False,
            }

        except Exception as e:

            failure_record = _build_failure_record(name, e)
            failures.append(failure_record)

            results[name] = {
                "status":         "failed",
                "error":          str(e),
                "exception_type": type(e).__name__,
            }

    # ── Attach failure log and engine metadata ───────────────────────────────
    if failures:
        results["_failures"] = failures

    results["_engine"] = {
        "engine_version":    ENGINE_VERSION,
        "observation_count": observation_count,
        "horizon":           horizon,
        "confidence_level":  confidence_level,
        "models_attempted":  len(get_model_registry()),
        "models_succeeded":  sum(
            1 for k, v in results.items()
            if not k.startswith("_") and isinstance(v, dict)
            and v.get("status") == "success"
        ),
    }

    return results