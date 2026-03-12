# FILE: analysis/forecast_runner.py
# ROLE: GOVERNED MODEL EXECUTION ORCHESTRATOR + EXECUTIVE ASSESSMENT
# STATUS: EXECUTIVE-GRADE / HARDENED
# ==================================================

from __future__ import annotations

import os
import uuid
import traceback
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd

from streamlit_sandbox.models.registry import get_model_registry
from streamlit_sandbox.models.contracts import ForecastResult

from analysis.diagnostics import compute_diagnostics


# ==================================================
# FAILURE LOGGING (APPEND-ONLY, NON-BLOCKING)
# ==================================================

def _log_model_failure(
    run_id: str,
    model_name: str,
    exc: Exception,
    context: dict | None = None,
) -> None:
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "forecast_failures.log")

        ts = datetime.now(timezone.utc).isoformat()
        ctx = context or {}

        record = (
            f"\n{'=' * 80}\n"
            f"ts_utc={ts}\n"
            f"run_id={run_id}\n"
            f"model={model_name}\n"
            f"error_type={type(exc).__name__}\n"
            f"error={str(exc)}\n"
            f"context={ctx}\n"
            f"traceback:\n{traceback.format_exc()}\n"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(record)

    except Exception:
        pass


# ==================================================
# ACTUALS ALIGNMENT (CANONICAL)
# ==================================================

def _inject_actuals(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    hist = historical_df[["date", "value"]].rename(columns={"value": "actual"})

    merged = pd.merge(
        forecast_df,
        hist,
        on="date",
        how="left",
        suffixes=("", "_hist"),
    )

    if "actual_hist" in merged.columns:
        merged["actual"] = merged["actual"].combine_first(merged["actual_hist"])
        merged = merged.drop(columns=["actual_hist"])

    return merged


# ==================================================
# EXECUTIVE ASSESSMENT (DIAGNOSTICS-FIRST)
# ==================================================

def _assess_executive_posture(
    metrics: Dict[str, float],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:

    risk_flags = []
    confidence = "Moderate"
    decision_use = "Directional guidance"

    # --------------------------------------------------
    # HARD GATES â€” NO EXCEPTIONS
    # --------------------------------------------------
    n_obs = diagnostics.get("n_observations", 0)
    ci_hit = diagnostics.get("ci_coverage")
    stability = diagnostics.get("stability_score")
    regime = diagnostics.get("regime_flag")

    if n_obs < 12:
        return {
            "confidence_posture": "Low",
            "decision_use": "Exploratory only",
            "risk_flags": ["Insufficient historical data"],
        }

    if ci_hit is None:
        return {
            "confidence_posture": "Low",
            "decision_use": "Exploratory only",
            "risk_flags": ["Unvalidated confidence intervals"],
        }

    if stability is None:
        return {
            "confidence_posture": "Low",
            "decision_use": "Exploratory only",
            "risk_flags": ["Unstable or insufficient error history"],
        }

    if regime == "Possible structural change":
        return {
            "confidence_posture": "Low",
            "decision_use": "Monitor only",
            "risk_flags": ["Potential regime instability"],
        }

    # --------------------------------------------------
    # ACCURACY (ONLY AFTER DIAGNOSTICS PASS)
    # --------------------------------------------------
    mape = metrics.get("MAPE")
    bias = metrics.get("Bias")
    mae = metrics.get("MAE", 0)

    if mape is not None:
        if mape < 0.10:
            confidence = "High"
            decision_use = "Operational planning / budgeting"
        elif mape < 0.20:
            confidence = "Moderate"
            decision_use = "Directional planning"
        else:
            confidence = "Low"
            decision_use = "Exploratory only"
            risk_flags.append("High forecast error")

    # --------------------------------------------------
    # BIAS RISK
    # --------------------------------------------------
    if bias is not None and abs(bias) > (mae * 0.5):
        risk_flags.append("Material forecast bias")

    # --------------------------------------------------
    # CI SHAPE RISK
    # --------------------------------------------------
    if ci_hit < 0.6:
        risk_flags.append("Under-calibrated confidence intervals")
    elif ci_hit > 0.95:
        risk_flags.append("Overly wide confidence intervals")

    return {
        "confidence_posture": confidence,
        "decision_use": decision_use,
        "risk_flags": risk_flags,
    }


# ==================================================
# FORECAST EXECUTION (DETERMINISTIC)
# ==================================================

def run_all_models(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> Dict[str, Any]:

    run_id = str(uuid.uuid4())
    results: Dict[str, Any] = {}

    hist_df = df.copy()
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df.sort_values("date")

    for model in get_model_registry():
        name = model["name"]
        runner = model["runner"]

        try:
            output = runner(
                df=hist_df,
                horizon=horizon,
                confidence_level=confidence_level,
            )

            if not isinstance(output, ForecastResult):
                raise TypeError(f"{name} did not return ForecastResult")

            forecast_df = output.forecast_df.copy()
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            forecast_df = _inject_actuals(
                forecast_df=forecast_df,
                historical_df=hist_df,
            )

            diagnostics = compute_diagnostics(
                forecast_df=forecast_df,
                metrics=output.metrics,
            )

            executive_assessment = _assess_executive_posture(
                metrics=output.metrics,
                diagnostics=diagnostics,
            )

            results[name] = {
                "status": "success",
                "forecast_df": forecast_df,
                "metrics": output.metrics,
                "diagnostics": diagnostics,
                "executive_assessment": executive_assessment,
                "metadata": {
                    **(output.metadata or {}),
                    "run_id": run_id,
                },
            }

        except Exception as e:
            _log_model_failure(
                run_id,
                name,
                e,
                {
                    "horizon": horizon,
                    "confidence_level": confidence_level,
                    "rows": int(len(hist_df)),
                    "columns": list(hist_df.columns),
                },
            )

            results[name] = {
                "status": "failed",
                "error": str(e),
                "run_id": run_id,
            }

    return results

