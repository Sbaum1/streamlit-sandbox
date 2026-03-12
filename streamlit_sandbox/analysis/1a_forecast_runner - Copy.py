# ==================================================
# FILE: analysis/forecast_runner.py
# ROLE: MODEL EXECUTION ORCHESTRATOR
# STATUS: CERTIFICATION-INTEGRATED / FORTUNE 100 HARDENED
# ==================================================

from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import traceback
import json
from datetime import datetime

from models.registry import get_model_registry
from models.contracts import ForecastResult
from analysis.diagnostics import compute_diagnostics
from analysis.backtest_engine import run_backtest, MIN_OBSERVATIONS


FAILURE_LOG_PATH = "forecast_failures.jsonl"


# ==================================================
# FAILURE LOGGER (NON-BLOCKING / APPEND ONLY)
# ==================================================

def _log_failure(model_name: str, error: Exception) -> dict:

    failure_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "exception_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }

    try:
        with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(failure_record) + "\n")
    except Exception:
        pass

    return failure_record


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

def _inject_actuals(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:

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
            raise ValueError(
                "Forecast horizon overlaps historical period."
            )


# ==================================================
# READINESS TIER LOGIC
# ==================================================

def _assign_readiness(metrics: dict) -> str:

    if not metrics.get("eligible"):
        return "Ineligible — Minimum Data Not Met"

    mase = metrics.get("mase")
    theils_u = metrics.get("theils_u")

    if mase is None or theils_u is None:
        return "Unscored"

    if mase < 0.8 and theils_u < 1.0:
        return "Tier 1 — Production Ready"

    if mase < 1.0:
        return "Tier 2 — Acceptable"

    return "Tier 3 — Weak"


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
                raise TypeError(
                    f"{name} did not return ForecastResult."
                )

            forecast_df = output.forecast_df.copy()
            forecast_df["date"] = pd.to_datetime(
                forecast_df["date"]
            )

            # --------------------------------------------------
            # DIAGNOSTIC-ONLY MODELS
            # --------------------------------------------------

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

            # --------------------------------------------------
            # ACTUALS + FORWARD VALIDATION
            # --------------------------------------------------

            forecast_df = _inject_actuals(
                forecast_df,
                hist_df,
            )

            _validate_forward_boundary(forecast_df)

            # --------------------------------------------------
            # CERTIFICATION BACKTEST
            # --------------------------------------------------

            if observation_count < MIN_OBSERVATIONS:

                metrics = {
                    "eligible": False,
                    "reason": (
                        f"Minimum {MIN_OBSERVATIONS} "
                        "observations required for certification."
                    ),
                    "observations": observation_count,
                }

            else:

                metrics = run_backtest(
                    df=hist_df,
                    model_runner=runner,
                    horizon=horizon,
                    confidence_level=confidence_level,
                )

                if not isinstance(metrics, dict):
                    raise ValueError(
                        "Backtest engine did not return dict."
                    )

                metrics["eligible"] = True
                metrics["observations"] = observation_count

            readiness = _assign_readiness(metrics)

            diagnostics = compute_diagnostics(
                forecast_df=forecast_df,
                metrics=metrics,
            )

            results[name] = {
                "status": "success",
                "forecast_df": forecast_df,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "executive_assessment": {
                    "readiness_tier": readiness,
                },
                "metadata": output.metadata or {},
            }

        except Exception as e:

            failure_record = _log_failure(name, e)
            failures.append(failure_record)

            results[name] = {
                "status": "failed",
                "error": str(e),
                "exception_type": type(e).__name__,
            }

    if failures:
        results["_failures"] = failures

    return results
