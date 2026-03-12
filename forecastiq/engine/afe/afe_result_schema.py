# ============================================================
# FILE: afe_result_schema.py
# ROLE: CANONICAL AFE EXECUTION OUTPUT SCHEMA
# STATUS: CANONICAL / GOVERNANCE-LOCKED
# ============================================================

from dataclasses import dataclass
from typing import Dict, List, Optional


# ------------------------------------------------------------
# FORECAST OUTPUT STRUCTURES
# ------------------------------------------------------------

@dataclass(frozen=True)
class ForecastInterval:
    base: List[float]
    upside: List[float]
    downside: List[float]


@dataclass(frozen=True)
class ForecastOutput:
    horizon: int
    point_forecast: List[float]
    intervals: Optional[ForecastInterval]


# ------------------------------------------------------------
# STRUCTURAL / DIAGNOSTIC OUTPUT
# ------------------------------------------------------------

@dataclass(frozen=True)
class StructuralOutput:
    signals: Dict[str, float]
    narrative: str


# ------------------------------------------------------------
# EXECUTION METADATA
# ------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionMetadata:
    model_id: str
    execution_mode: str  # Qualified | Force Executed
    dataset_hash: str
    executed_at: str
    parameter_snapshot: Dict[str, str]


# ------------------------------------------------------------
# CANONICAL AFE RESULT
# ------------------------------------------------------------

@dataclass(frozen=True)
class AFEResult:
    metadata: ExecutionMetadata
    forecast: Optional[ForecastOutput]
    structure: Optional[StructuralOutput]
    limitations: str

    # --------------------------------------------------------
    # GOVERNED SEMANTIC SIGNALS (READ-ONLY)
    # --------------------------------------------------------

    @property
    def is_force_executed(self) -> bool:
        """
        Explicit semantic indicator for Force Executed models.
        This must be used by all downstream consumers (UI, export,
        audit, reporting) instead of inferring from strings.
        """
        return self.metadata.execution_mode == "Force Executed"
