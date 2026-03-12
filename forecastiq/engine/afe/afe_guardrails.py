# ============================================================
# FILE: afe_guardrails.py
# ROLE: AFE EXECUTION MODES & GOVERNANCE GUARDRAILS
# STATUS: CANONICAL / GOVERNANCE-LOCKED
# ============================================================

from typing import List

from forecastiq.engine.afe.afe_contract import (
    AFEExecutionInput,
    AFEModelSuitability,
)


# ------------------------------------------------------------
# CONSTANTS (LOCKED)
# ------------------------------------------------------------

STRONG_FIT = "Strong Fit"
CONDITIONAL_FIT = "Conditional Fit"
NOT_RECOMMENDED = "Not Recommended"


# ------------------------------------------------------------
# QUALIFIED EXECUTION RESOLUTION
# ------------------------------------------------------------

def resolve_executable_models(input_bundle: AFEExecutionInput) -> List[str]:
    """
    Determines models eligible for Qualified Execution under AFE rules.
    """

    executable_models: List[str] = []

    for model_id, suitability in input_bundle.suitability.items():
        if not isinstance(suitability, AFEModelSuitability):
            raise TypeError(
                f"Invalid suitability object for model '{model_id}'."
            )

        if suitability.classification in (STRONG_FIT, CONDITIONAL_FIT):
            executable_models.append(model_id)

    return executable_models


# ------------------------------------------------------------
# FORCE EXECUTION VALIDATION
# ------------------------------------------------------------

def validate_force_execution(input_bundle: AFEExecutionInput) -> List[str]:
    """
    Validates and returns models explicitly authorized for Force Execution.
    """

    auth = input_bundle.authorization

    if not auth.allow_force_execution:
        return []

    if not auth.force_models:
        raise ValueError(
            "Force execution enabled but no models explicitly specified."
        )

    for model_id in auth.force_models:
        if model_id not in input_bundle.suitability:
            raise ValueError(
                f"Force execution requested for unknown model '{model_id}'."
            )

    return auth.force_models


# ------------------------------------------------------------
# FINAL EXECUTION PLAN RESOLUTION
# ------------------------------------------------------------

def resolve_execution_plan(input_bundle: AFEExecutionInput) -> List[str]:
    """
    Produces the final list of models to execute, enforcing all AFE
    governance constraints.
    """

    qualified_models = resolve_executable_models(input_bundle)
    force_models = validate_force_execution(input_bundle)

    execution_plan = set(qualified_models)

    for model_id in force_models:
        execution_plan.add(model_id)

    if not execution_plan:
        raise ValueError(
            "AFE execution aborted: no models authorized for execution."
        )

    return sorted(execution_plan)
