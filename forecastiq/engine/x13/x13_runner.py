# ============================================================
# FILE: x13_runner.py
# ROLE: X-13 ARIMA-SEATS EXECUTION RUNNER (GOVERNED)
# ============================================================

from typing import Dict
from forecastiq.engine.x13.x13_contract import X13ContractResult
from forecastiq.engine.x13.x13_artifacts import store_census_artifacts


def run_x13(
    *,
    qualified: bool,
    force_executed: bool,
    dataset_hash: str,
    spec_hash: str,
) -> X13ContractResult:
    """
    Executes X-13 ARIMA-SEATS if qualified or force executed.

    NOTE:
    - Census binary execution is intentionally stubbed
    - No exceptions propagate into AFE
    - Failure is reported via contract only
    """

    if not qualified and not force_executed:
        return X13ContractResult(
            executed=False,
            qualified=qualified,
            force_executed=force_executed,
            seasonality_confirmed=False,
            seasonal_stability=None,
            residual_quality=None,
            artifact_id=None,
            failure_reason=None,
        )

    try:
        # ----------------------------------------------------
        # PLACEHOLDER: Census X-13 binary execution
        # ----------------------------------------------------
        # Replace this block in Phase 2 with real Census calls

        raw_outputs: Dict[str, bytes] = {
            "x13.out": b"X-13 ARIMA-SEATS OUTPUT (STUB)",
            "x13.d11": b"SEASONALLY ADJUSTED SERIES (STUB)",
            "x13.d12": b"TREND CYCLE (STUB)",
            "x13.d13": b"IRREGULAR (STUB)",
        }

        artifact_id = store_census_artifacts(
            raw_outputs=raw_outputs,
            dataset_hash=dataset_hash,
            spec_hash=spec_hash,
            census_version="CENSUS-X13-STUB",
        )

        return X13ContractResult(
            executed=True,
            qualified=qualified,
            force_executed=force_executed,
            seasonality_confirmed=True,
            seasonal_stability="medium",
            residual_quality="acceptable",
            artifact_id=artifact_id,
            failure_reason=None,
        )

    except Exception as exc:
        return X13ContractResult(
            executed=False,
            qualified=qualified,
            force_executed=force_executed,
            seasonality_confirmed=False,
            seasonal_stability=None,
            residual_quality="failed",
            artifact_id=None,
            failure_reason=str(exc),
        )
