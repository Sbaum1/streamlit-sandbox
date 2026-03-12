# ==================================================
# FILE: streamlit_sandbox/execution/engine.py
# ROLE: MODEL EXECUTION ENGINE (DETERMINISTIC CLASSIFICATION HARDENED)
# STANDARD: FORTUNE 100 / ZERO REGRESSION
# ==================================================

from __future__ import annotations

from typing import Dict, Callable

from streamlit_sandbox.models.arima import run_arima
from streamlit_sandbox.models.sarima import run_sarima
from streamlit_sandbox.models.sarimax import run_sarimax
from streamlit_sandbox.models.theta import run_theta
from streamlit_sandbox.models.stl_ets import run_stl_ets
from streamlit_sandbox.models.tbats import run_tbats
from streamlit_sandbox.models.prophet_model import run_prophet
from streamlit_sandbox.models.bsts import run_bsts
from streamlit_sandbox.models.x13 import run_x13
from streamlit_sandbox.models.primary_ensemble import run_primary_ensemble


# --------------------------------------------------
# MODEL REGISTRY (ORDER PRESERVED)
# --------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable] = {
    "ARIMA": run_arima,
    "SARIMA": run_sarima,
    "SARIMAX": run_sarimax,
    "Theta": run_theta,
    "STL+ETS": run_stl_ets,
    "TBATS": run_tbats,
    "Prophet": run_prophet,
    "BSTS": run_bsts,
    "X-13": run_x13,
    "Primary Ensemble": run_primary_ensemble,
}


# --------------------------------------------------
# CORE EXECUTION
# --------------------------------------------------

def run_all_models(
    df,
    horizon: int,
    confidence_level: float,
):

    results: Dict[str, dict] = {}
    failures: Dict[str, str] = {}

    for model_name in MODEL_REGISTRY:

        runner = MODEL_REGISTRY[model_name]

        try:
            forecast_result = runner(
                df=df,
                horizon=horizon,
                confidence_level=confidence_level,
            )

            forecast_df = getattr(forecast_result, "forecast_df", None)
            metadata = getattr(forecast_result, "metadata", {}) or {}

            empty_output = False

            if forecast_df is None or len(forecast_df) == 0:
                empty_output = True

            results[model_name] = {
                "model_name": model_name,
                "status": "success",
                "forecast_df": forecast_df,
                "metadata": metadata,
                "empty_output": empty_output,
            }

        except Exception as e:

            error_metadata = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_deterministic": True,
            }

            results[model_name] = {
                "model_name": model_name,
                "status": "error",
                "forecast_df": None,
                "metadata": error_metadata,
                "empty_output": False,
            }

            failures[model_name] = str(e)

    results["_failures"] = failures

    return results