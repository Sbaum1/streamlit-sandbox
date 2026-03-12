# ==================================================
# FILE: streamlit_sandbox/analysis/forecast_runner.py
# ROLE: MODEL EXECUTION ORCHESTRATOR
# STATUS: CERTIFICATION-INTEGRATED / FORTUNE 100 HARDENED / EXECUTIVE-READY
# ==================================================

from __future__ import annotations

from typing import Dict, Any
import pandas as pd

from streamlit_sandbox.models.registry import get_model_registry
from streamlit_sandbox.models.contracts import ForecastResult
from streamlit_sandbox.analysis.diagnostics import compute_diagnostics
from streamlit_sandbox.analysis.backtest_engine import run_backtest, MIN_OBSERVATIONS


# ==================================================
# FAILURE RECORD (PURE / DETERMINISTIC / NO SIDE EFFECTS)
# ==================================================

def _build_failure_record(model_name: str, error: Exception) -> dict:
    return {
        "model_name": model_name,
        "error_type": type(error).__name__,
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

    df = df.sort_values("date")

    inferred = pd.infer_freq(df["date"])
    if inferred not in ("MS", "M"):
        raise ValueError("Monthly frequency required.")

    df = df.set_index("date").asfreq(inferred).reset_index()

    if df["value"].isna().any():
        raise ValueError("Missing values detected after frequency alignment.")

    return df


# ==================================================
# ACTUALS INJECTION
# ==================================================

def _inject_actuals(forecast_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:

    df = forecast_df.copy()

    if "actual" not in df.columns:
        df["actual"] = pd.NA

    hist = historical_df[["date", "value"]].rename(
        columns={"value": "actual_hist"}
    )

    merged = pd.merge(df, hist, on="date", how="left")
    merged["actual"] = merged["actual"].combine_first(
        merged["actual_hist"]
    )
    merged = merged.drop(columns=["actual_hist"])

    if merged["date"].duplicated().any():
        raise ValueError("Duplicate dates after actual injection.")

    return merged


# ==================================================
# FORWARD BOUNDARY CHECK
# ==================================================

def _validate_forward_boundary(forecast_df: pd.DataFrame):

    df = forecast_df.sort_values("date")

    hist = df[df["actual"].notna()]
    future = df[df["actual"].isna()]

    if not hist.empty and not future.empty:
        if future["date"].min() <= hist["date"].max():
            raise ValueError("Forecast horizon overlaps historical period.")


# ==================================================
# METRIC NORMALIZATION (AUTHORITATIVE CONTRACT)
# ==================================================

def _normalize_metric_keys(metrics: dict) -> dict:

    if not isinstance(metrics, dict):
        return {}

    mapping = {
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE",
        "bias": "Bias",
        "mase": "MASE",
        "theils_u": "Theils_U",
        "ci_coverage": "CI_Coverage",
        "smape": "SMAPE",
        "folds": "Folds",
        "observations": "Observations",
    }

    normalized = {}

    for k, v in metrics.items():
        normalized[mapping.get(k, k)] = v

    return normalized


# ==================================================
# READINESS TIER ENGINE (EXECUTIVE HARDENED)
# ==================================================

def _assign_readiness(metrics: dict, confidence_level: float) -> str:

    if metrics.get("eligible") is False:
        return "Ineligible — Minimum Data Not Met"

    mase = metrics.get("MASE")
    theils_u = metrics.get("Theils_U")
    coverage = metrics.get("CI_Coverage")
    bias = metrics.get("Bias")
    folds = metrics.get("Folds")
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
# EXECUTIVE CONFIDENCE ENGINE (CALIBRATED + RISK AWARE)
# ==================================================

def _assign_confidence(metrics: dict, confidence_level: float) -> dict:

    if metrics.get("eligible") is False:
        return {
            "confidence_posture": "Not Eligible",
            "risk_flags": ["Minimum data threshold not met"],
            "decision_guidance": "Increase historical data before executive use.",
        }

    mase = metrics.get("MASE")
    theils_u = metrics.get("Theils_U")
    coverage = metrics.get("CI_Coverage")
    bias = metrics.get("Bias")
    mean_level = metrics.get("Mean_Level")

    risk_flags = []

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
            posture = "High Confidence — Production Safe"
            guidance = "Model suitable for executive planning use."

        elif mase < 1.0:
            posture = "Moderate Confidence — Monitor"
            guidance = "Model acceptable but monitor volatility and calibration."

        else:
            posture = "Low Confidence — Elevated Risk"
            guidance = "Exercise caution in executive decisions."

    else:
        posture = "Unscored"
        guidance = "Metrics incomplete."

    return {
        "confidence_posture": posture,
        "risk_flags": risk_flags,
        "decision_guidance": guidance,
    }


# ==================================================
# MAIN ROUTER
# ==================================================

def run_all_models(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}
    failures: list = []

    hist_df = _validate_input(df)
    observation_count = len(hist_df)

    for model_meta in get_model_registry():

        name = model_meta["name"]
        runner = model_meta["runner"]
        diagnostic_only = model_meta.get("diagnostic_only", False)

        try:

            output = runner(
                df=hist_df,
                horizon=horizon,
                confidence_level=confidence_level,
            )

            if not isinstance(output, ForecastResult):
                raise TypeError(f"{name} did not return ForecastResult.")

            forecast_df = output.forecast_df.copy()
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            if diagnostic_only:
                results[name] = {
                    "status": "success",
                    "forecast_df": forecast_df,
                    "metrics": {},
                    "diagnostics": {},
                    "executive_assessment": {},
                    "metadata": output.metadata or {},
                }
                continue

            forecast_df = _inject_actuals(forecast_df, hist_df)
            _validate_forward_boundary(forecast_df)

            if observation_count < MIN_OBSERVATIONS:

                raw_metrics = {
                    "eligible": False,
                    "reason": f"Minimum {MIN_OBSERVATIONS} observations required.",
                    "observations": observation_count,
                }

            else:

                raw_metrics = run_backtest(
                    df=hist_df,
                    model_runner=runner,
                    horizon=horizon,
                    confidence_level=confidence_level,
                )

                if not isinstance(raw_metrics, dict):
                    raise ValueError("Backtest engine did not return dict.")

                raw_metrics["eligible"] = True
                raw_metrics["observations"] = observation_count

            normalized_metrics = _normalize_metric_keys(raw_metrics)

            readiness = _assign_readiness(
                normalized_metrics,
                confidence_level,
            )

            confidence_bundle = _assign_confidence(
                normalized_metrics,
                confidence_level,
            )

            diagnostics = compute_diagnostics(
                forecast_df=forecast_df,
                metrics=normalized_metrics,
            )

            metadata = output.metadata or {}
            metadata["confidence_level"] = confidence_level

            results[name] = {
                "status": "success",
                "forecast_df": forecast_df,
                "metrics": normalized_metrics,
                "diagnostics": diagnostics,
                "executive_assessment": {
                    "readiness_tier": readiness,
                    **confidence_bundle,
                },
                "metadata": metadata,
            }

        except Exception as e:

            failure_record = _build_failure_record(name, e)
            failures.append(failure_record)

            results[name] = {
                "status": "failed",
                "error": str(e),
                "exception_type": type(e).__name__,
            }

    if failures:
        results["_failures"] = failures

    return results