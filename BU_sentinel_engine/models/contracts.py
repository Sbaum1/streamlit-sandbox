# FILE: streamlit_sandbox/models/contracts.py
# ROLE: FORECAST EXECUTION CONTRACT
# PURPOSE: Canonical, immutable contract for all forecast model outputs,
#          including transparency metadata for readiness and confidence intervals.
# STATUS: EXECUTIVE-GRADE / CANONICAL
# ==================================================

from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any


@dataclass(frozen=True)
class ForecastResult:
    """
    Canonical forecast result contract.

    REQUIRED:
    - forecast_df: full history + future rows
    - metrics: accuracy metrics computed on realized data only
    - metadata: transparency-only descriptors (NO ranking logic)

    METADATA CONVENTIONS (OPTIONAL, NON-BINDING):
    - readiness_tier: 'Primary' | 'Secondary' | 'Operational Only'
    - ci_note: explicit disclosure when confidence intervals are placeholders
    - warnings / notes: execution or interpretability caveats
    """

    model_name: str
    forecast_df: pd.DataFrame   # date, actual, forecast, ci_low, ci_mid, ci_high, error_pct
    metrics: Dict[str, float]   # MAE, RMSE, MAPE, Bias (where available)
    metadata: Dict[str, Any]    # transparency-only descriptors (no logic)