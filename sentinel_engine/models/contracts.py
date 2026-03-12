# ==================================================
# FILE: sentinel_engine/models/contracts.py
# VERSION: 2.0.0
# ROLE: CONTRACT RE-EXPORT SHIM
# ENGINE: Sentinel Engine v2.0.0
# ==================================================
#
# PURPOSE:
#   This file exists solely to ensure that model files
#   importing ForecastResult from this local path receive
#   the SAME class object as sentinel_engine.contracts.
#
#   Without this shim, Python creates two separate class
#   objects for ForecastResult — one per import path —
#   causing isinstance() checks in runner.py to fail even
#   though the classes are structurally identical.
#
# GOVERNANCE:
#   - Never define ForecastResult here directly
#   - Always re-export from sentinel_engine.contracts
#   - This file must never contain any logic
# ==================================================

from sentinel_engine.contracts import ForecastResult, ENGINE_VERSION

__all__ = ["ForecastResult", "ENGINE_VERSION"]