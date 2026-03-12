# ============================================================
# FILE: afe_orchestrator.py
# ROLE: ADVANCED FORECAST EXECUTION ENGINE (AFE) ORCHESTRATOR
# STATUS: CANONICAL / GOVERNANCE-LOCKED / FINAL
# ============================================================

from typing import List, Dict
from datetime import datetime

from forecastiq.engine.afe.afe_contract import AFEExecutionInput
from forecastiq.engine.afe.afe_guardrails import resolve_execution_plan
from forecastiq.engine.afe.afe_result_schema import (
    AFEResult,
    ExecutionMetadata,
)
from forecastiq.engine.afe.afe_audit import compute_dataset_hash
from forecastiq.engine.afe.afe_execution import execute_model

from forecastiq.engine.afe.models.naive_persistence import run_naive_persistence
from forecastiq.engine.afe.models.ets import run_ets
from forecastiq.engine.afe.models.arima import run_arima
from forecastiq.engine.afe.models.prophet import run_prophet
from forecastiq.engine.afe.models.sarima import run_sarima
from forecastiq.engine.afe.models.short_horizon_error_minimizing import (
    run_short_horizon_error_minimizing,
)
from forecastiq.engine.afe.models.bsts import run_bsts
from forecastiq.engine.afe.models.intermittent_sparse_demand import (
    run_intermittent_sparse_demand,
)
from forecastiq.engine.afe.models.ensemble_consensus_reference import (
    run_ensemble_consensus_reference,
)

# ------------------------------------------------------------
# GOVERNED MODEL IDENTIFIERS (LOCKED)
# ------------------------------------------------------------

MODEL_NAIVE = "Naïve / Persistence Model"
MODEL_ETS = "ETS (Error–Trend–Seasonality)"
MODEL_ARIMA = "ARIMA"
MODEL_PROPHET = "Prophet"
MODEL_SARIMA = "Seasonal ARIMA (SARIMA)"
MODEL_SHORT_HORIZON = "Short-Horizon Error-Minimizing Model"
MODEL_BSTS = "Bayesian Structural Time Series (BSTS)"
MODEL_INTERMITTENT = "Intermittent / Sparse Demand Model"
MODEL_ENSEMBLE = "Ensemble / Consensus Reference"
MODEL_X13 = "X-13 ARIMA-SEATS"

DEFAULT_FORECAST_HORIZON = 3

# ------------------------------------------------------------
# ORCHESTRATOR ENTRY POINT
# ------------------------------------------------------------

def run_afe(input_bundle: AFEExecutionInput) -> List[AFEResult]:
    """
    Executes the Advanced Forecast Execution Engine (AFE)
    under full governance constraints.
    """

    if input_bundle.dataset is None:
        raise ValueError("AFE execution aborted: committed dataset missing.")
    if input_bundle.intelligence is None:
        raise ValueError("AFE execution aborted: dataset intelligence missing.")
    if not input_bundle.suitability:
        raise ValueError("AFE execution aborted: model suitability missing.")
    if input_bundle.authorization is None:
        raise ValueError("AFE execution aborted: execution authorization missing.")

    models_to_execute = resolve_execution_plan(input_bundle)

    dataset_hash = compute_dataset_hash(input_bundle.dataset)
    execution_time = datetime.utcnow().isoformat()

    results: List[AFEResult] = []
    forecast_buffer: Dict[str, object] = {}

    # --------------------------------------------------------
    # PASS 1 — FORECAST & STRUCTURAL MODELS
    # --------------------------------------------------------

    for model_id in models_to_execute:

        if model_id == MODEL_ENSEMBLE:
            continue  # ensemble runs last

        qualified = (
            model_id in input_bundle.suitability
            and input_bundle.suitability[model_id].classification
            in ("Strong Fit", "Conditional Fit")
        )

        force_executed = (
            input_bundle.authorization.force_models
            and model_id in input_bundle.authorization.force_models
        )

        metadata = ExecutionMetadata(
            model_id=model_id,
            execution_mode="Force Executed" if force_executed else "Qualified",
            dataset_hash=dataset_hash,
            executed_at=execution_time,
            parameter_snapshot={},
        )

        forecast = None
        structure = None
        limitations = ""

        try:
            if model_id == MODEL_NAIVE:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_naive_persistence,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_ETS:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_ets,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_ARIMA:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_arima,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_PROPHET:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_prophet,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_SARIMA:
                period = (
                    input_bundle.overrides.sarima_seasonal_period
                    if input_bundle.overrides and input_bundle.overrides.sarima_seasonal_period
                    else input_bundle.intelligence.dominant_periods[0]
                )
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_sarima,
                    dataset=input_bundle.dataset,
                    intelligence=input_bundle.intelligence,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={
                        "horizon": DEFAULT_FORECAST_HORIZON,
                        "seasonal_period": period,
                    },
                )["output"]

            elif model_id == MODEL_SHORT_HORIZON:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_short_horizon_error_minimizing,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_BSTS:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_bsts,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_INTERMITTENT:
                forecast = execute_model(
                    model_name=model_id,
                    model_callable=run_intermittent_sparse_demand,
                    dataset=input_bundle.dataset,
                    intelligence=None,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"horizon": DEFAULT_FORECAST_HORIZON},
                )["output"]

            elif model_id == MODEL_X13:
                structure = execute_model(
                    model_name="x13_arima_seats",
                    model_callable=None,
                    dataset=input_bundle.dataset,
                    intelligence=input_bundle.intelligence,
                    qualified=qualified,
                    force_executed=force_executed,
                    execution_kwargs={"spec_hash": "LOCKED"},
                )["output"]

        except Exception as exc:
            limitations = f"Execution failed: {exc}"

        if forecast is not None:
            forecast_buffer[model_id] = forecast

        results.append(
            AFEResult(
                metadata=metadata,
                forecast=forecast,
                structure=structure,
                limitations=limitations,
            )
        )

    # --------------------------------------------------------
    # PASS 2 — ENSEMBLE (CONSENSUS)
    # --------------------------------------------------------

    if MODEL_ENSEMBLE in models_to_execute:
        metadata = ExecutionMetadata(
            model_id=MODEL_ENSEMBLE,
            execution_mode="Qualified",
            dataset_hash=dataset_hash,
            executed_at=execution_time,
            parameter_snapshot={},
        )

        try:
            structure = run_ensemble_consensus_reference(
                model_outputs=forecast_buffer
            )
            limitations = ""
        except Exception as exc:
            structure = None
            limitations = f"Execution failed: {exc}"

        results.append(
            AFEResult(
                metadata=metadata,
                forecast=None,
                structure=structure,
                limitations=limitations,
            )
        )

    return results
