# ============================================================
# FILE: afe_execution.py
# ROLE: AFE EXECUTION ENGINE (LOCKED)
# STATUS: CANONICAL / GOVERNED
# ============================================================

"""
AFE EXECUTION

Role:
- Execute individual forecast or diagnostic models
- Preserve determinism and original evaluation context
- Handle qualified execution and force execution identically
  except for explicit labeling

MUST:
- Execute models exactly as evaluated
- Preserve known weaknesses
- Emit raw, unadjusted outputs
- Respect qualification + force-execution gates

MUST NOT:
- Optimize, refit, smooth, or correct model behavior
- Introduce fallback or silent recovery
- Alter uncertainty assumptions
- Leak external authority artifacts into AFE state
"""

from typing import Dict, Any

from forecastiq.engine.afe.afe_contract import (
    AFECommittedDataset,
    AFEDatasetIntelligence,
)
from forecastiq.engine.afe.afe_result_schema import (
    ForecastOutput,
    StructuralOutput,
)

# X-13 external authority (diagnostic only)
from forecastiq.engine.x13.x13_runner import run_x13
from forecastiq.engine.x13.x13_contract import X13ContractResult


# ------------------------------------------------------------
# EXECUTION DISPATCH
# ------------------------------------------------------------

def execute_model(
    *,
    model_name: str,
    model_callable,
    dataset: AFECommittedDataset,
    intelligence: AFEDatasetIntelligence | None,
    qualified: bool,
    force_executed: bool,
    execution_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a single AFE model under strict governance.

    Returns a dictionary containing:
    - model_name
    - execution_mode
    - output (ForecastOutput | StructuralOutput | X13ContractResult)
    """

    execution_mode = (
        "FORCE_EXECUTED" if force_executed else "QUALIFIED_EXECUTION"
    )

    # --------------------------------------------------------
    # X-13 SPECIAL HANDLING (EXTERNAL AUTHORITY)
    # --------------------------------------------------------

    if model_name == "x13_arima_seats":
        x13_result: X13ContractResult = run_x13(
            qualified=qualified,
            force_executed=force_executed,
            dataset_hash=dataset.dataset_hash,
            spec_hash=execution_kwargs.get("spec_hash", "UNKNOWN"),
        )

        return {
            "model_name": model_name,
            "execution_mode": execution_mode,
            "output": x13_result,
        }

    # --------------------------------------------------------
    # STANDARD MODEL EXECUTION
    # --------------------------------------------------------

    if not qualified and not force_executed:
        raise RuntimeError(
            f"Model '{model_name}' attempted execution without "
            "qualification or force execution."
        )

    try:
        if intelligence is not None:
            output = model_callable(
                dataset=dataset,
                intelligence=intelligence,
                **execution_kwargs,
            )
        else:
            output = model_callable(
                dataset=dataset,
                **execution_kwargs,
            )

    except Exception as exc:
        raise RuntimeError(
            f"AFE execution failed for model '{model_name}': {exc}"
        ) from exc

    if not isinstance(output, (ForecastOutput, StructuralOutput)):
        raise TypeError(
            f"Model '{model_name}' returned invalid output type "
            f"{type(output)}."
        )

    return {
        "model_name": model_name,
        "execution_mode": execution_mode,
        "output": output,
    }
