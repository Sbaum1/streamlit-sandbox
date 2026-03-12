# ============================================================
# FILE: afe_contract.py
# ROLE: AFE INPUT & EXECUTION CONTRACT (IMMUTABLE)
# STATUS: CANONICAL / GOVERNANCE-LOCKED
# ============================================================

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


# ------------------------------------------------------------
# DATASET CONTRACT
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFECommittedDataset:
    dataset_id: str
    dataset_hash: str
    values: List[float]   # EXPLICIT NUMERIC SERIES
    row_count: int
    frequency: str
    start_date: str
    end_date: str
    committed_at: datetime


# ------------------------------------------------------------
# DATASET INTELLIGENCE CONTRACT
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFEDatasetIntelligence:
    observation_count: int
    missing_ratio: float
    sparsity_detected: bool
    seasonality_detected: bool
    seasonality_strength: Optional[str]
    dominant_periods: Optional[List[int]]
    long_term_cagr: Optional[float]
    recent_growth_rate: Optional[float]
    volatility_regime: str
    structural_instability_flag: bool


# ------------------------------------------------------------
# MODEL SUITABILITY CONTRACT
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFEModelSuitability:
    model_id: str
    classification: str  # Strong Fit | Conditional Fit | Not Recommended
    rationale: str
    known_limitations: List[str]


# ------------------------------------------------------------
# EXECUTION AUTHORIZATION CONTRACT
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFEExecutionAuthorization:
    allow_force_execution: bool
    force_models: Optional[List[str]]
    authorized_by: str
    authorized_at: datetime


# ------------------------------------------------------------
# EXECUTIVE EXECUTION OVERRIDES (OPTIONAL, GOVERNED)
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFEExecutionOverrides:
    """
    Explicit executive execution overrides.

    Governance Rules:
    - All fields are OPTIONAL and DEFAULT TO NONE
    - No inference, fallback, or auto-population is permitted
    - Overrides do NOT relax suitability or authorization rules
    - Overrides apply ONLY to explicitly supported models
    - Overrides are auditable and must be surfaced verbatim downstream
    """

    # SARIMA ONLY:
    # Explicit seasonal period to use when SARIMA execution is permitted.
    # Must EXACTLY match a dominant period detected by Dataset Intelligence.
    sarima_seasonal_period: Optional[int] = None


# ------------------------------------------------------------
# AFE INPUT BUNDLE (SINGLE ENTRY POINT)
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFEExecutionInput:
    dataset: AFECommittedDataset
    intelligence: AFEDatasetIntelligence
    suitability: Dict[str, AFEModelSuitability]
    authorization: AFEExecutionAuthorization
    overrides: Optional[AFEExecutionOverrides] = None
