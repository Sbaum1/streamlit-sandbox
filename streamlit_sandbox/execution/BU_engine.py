# FILE: streamlit_sandbox/execution/engine.py
# ROLE: EXECUTION FACADE (DELEGATES TO CANONICAL RUNNER)
# STATUS: GOVERNED
# ==================================================

import pandas as pd
from typing import Dict, Any

from streamlit_sandbox.analysis.forecast_runner import run_all_models as _run_all_models


def run_all_models(
    df: pd.DataFrame,
    horizon: int,
    confidence_level: float,
) -> Dict[str, Any]:
    """
    Delegates to canonical forecast runner.

    Governance:
    - Single execution path
    - Dict-based results keyed by model name
    - No transformation, no filtering
    """

    return _run_all_models(
        df=df,
        horizon=horizon,
        confidence_level=confidence_level,
    )