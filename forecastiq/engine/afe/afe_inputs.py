"""
AFE Input Contract — Sentinel v4
--------------------------------
Defines the ONLY allowed inputs to the Advanced Forecast Execution Engine.

This module contains:
- No execution logic
- No model logic
- No Streamlit
- No side effects

Purpose:
- Enforce determinism
- Preserve auditability
- Prevent implicit execution
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass(frozen=True)
class AFEExecutionAuthorization:
    """
    Explicit authorization for execution.
    """
    qualified_execution: bool = True
    force_execute_models: Optional[List[str]] = None


@dataclass(frozen=True)
class AFEDatasetContext:
    """
    Immutable dataset context.
    """
    dataset_hash: str
    row_count: int
    frequency: str
    start_date: datetime
    end_date: datetime


@dataclass(frozen=True)
class AFEDatasetIntelligence:
    """
    Outputs from Dataset Intelligence phase.
    """
    seasonality_strength: str
    trend_strength: str
    volatility_profile: str
    intermittency_flag: bool
    growth_context: Dict[str, Any]


@dataclass(frozen=True)
class AFEModelSuitability:
    """
    Suitability classification per model.
    """
    model_id: str
    classification: str  # Strong Fit | Conditional Fit | Not Recommended
    limitations: List[str]


@dataclass(frozen=True)
class AFEInputBundle:
    """
    Canonical input bundle for AFE execution.
    """
    dataset_context: AFEDatasetContext
    dataset_intelligence: AFEDatasetIntelligence
    model_suitability: List[AFEModelSuitability]
    execution_authorization: AFEExecutionAuthorization
