# ============================================================
# FILE: x13_contract.py
# ROLE: X-13 AFE OUTPUT CONTRACT (LOCKED)
# ============================================================

from dataclasses import dataclass
from typing import Optional, Literal


SeasonalStability = Literal["low", "medium", "high"]
ResidualQuality = Literal["acceptable", "noisy", "failed"]


@dataclass(frozen=True)
class X13ContractResult:
    executed: bool
    qualified: bool
    force_executed: bool

    seasonality_confirmed: bool
    seasonal_stability: Optional[SeasonalStability]
    residual_quality: Optional[ResidualQuality]

    artifact_id: Optional[str]
    failure_reason: Optional[str]
