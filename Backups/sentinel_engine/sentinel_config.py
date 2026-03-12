# ==================================================
# FILE: sentinel_engine/sentinel_config.py
# VERSION: 1.0.0
# ROLE: FEATURE FLAG AND TIER CONFIGURATION
# ENGINE: Sentinel Engine v2.0.0
# PHASE: 3F — Feature Flag Architecture
# ==================================================
#
# GOVERNANCE:
# - No Streamlit dependencies
# - No session state dependencies
# - Single source of truth for platform tier and feature flags
# - Importable by runner.py, ensemble.py, and app layer
# - All mutations go through set_tier() or set_flag()
# - Direct attribute mutation is permitted but discouraged
#
# TIER HIERARCHY:
#   essentials  — 10 models, 8 ensemble members
#   pro         — 13 models, 10 ensemble members
#   enterprise  — 19 models, 12 ensemble members (default)
#
# FEATURE FLAGS:
#   BACKTEST_ENABLED          — if False, runner skips backtest
#                               and ensemble uses equal weights.
#                               Use for fast dev/debug runs.
#
#   MASE_EXCLUSION_ENABLED    — if False, Phase 3D auto-exclusion
#                               gate (MASE > 2.0) is bypassed.
#                               All models included in ensemble
#                               regardless of MASE score.
#
#   DIVERSITY_CAP_ENABLED     — if False, ARIMA family cap (40%)
#                               is not applied. Equal or MASE
#                               weights used without redistribution.
#
#   INTERMITTENT_ROUTING_ENABLED — if False, Croston routing is
#                               bypassed. Intermittent series run
#                               through standard ensemble.
#
# USAGE:
#   from sentinel_engine.sentinel_config import get_config, set_tier
#
#   config = get_config()
#   print(config.ACTIVE_TIER)         # "enterprise"
#   print(config.BACKTEST_ENABLED)    # True
#
#   set_tier("pro")                   # Switch to Pro tier
#   set_flag("BACKTEST_ENABLED", False)  # Disable backtest
#
# APP LAYER (Streamlit):
#   Call set_tier(tier) at session start based on user plan.
#   Config is module-level — shared across all calls in process.
#   For multi-tenant isolation, use per-request config injection
#   via run_all_models(config_override=...) in Phase 4.
# ==================================================

from __future__ import annotations

from typing import List, Optional

# --------------------------------------------------
# VALID VALUES
# --------------------------------------------------

VALID_TIERS = {"essentials", "pro", "enterprise"}

VALID_FLAGS = {
    "BACKTEST_ENABLED",
    "MASE_EXCLUSION_ENABLED",
    "DIVERSITY_CAP_ENABLED",
    "INTERMITTENT_ROUTING_ENABLED",
}


# --------------------------------------------------
# CONFIG CLASS
# --------------------------------------------------

class SentinelConfig:
    """
    Platform configuration for Sentinel Engine.

    Holds the active tier and all feature flags.
    Constructed once at module load. Mutated via
    set_tier() and set_flag() accessors.
    """

    def __init__(
        self,
        active_tier:                   str  = "enterprise",
        ensemble_members_override:     Optional[List[str]] = None,
        backtest_enabled:              bool = True,
        mase_exclusion_enabled:        bool = True,
        diversity_cap_enabled:         bool = True,
        intermittent_routing_enabled:  bool = True,
    ) -> None:
        self.ACTIVE_TIER:                   str              = active_tier
        self.ENSEMBLE_MEMBERS_OVERRIDE:     Optional[List[str]] = ensemble_members_override
        self.BACKTEST_ENABLED:              bool             = backtest_enabled
        self.MASE_EXCLUSION_ENABLED:        bool             = mase_exclusion_enabled
        self.DIVERSITY_CAP_ENABLED:         bool             = diversity_cap_enabled
        self.INTERMITTENT_ROUTING_ENABLED:  bool             = intermittent_routing_enabled

    def as_dict(self) -> dict:
        """Return config as plain dict for metadata logging."""
        return {
            "active_tier":                  self.ACTIVE_TIER,
            "ensemble_members_override":    self.ENSEMBLE_MEMBERS_OVERRIDE,
            "backtest_enabled":             self.BACKTEST_ENABLED,
            "mase_exclusion_enabled":       self.MASE_EXCLUSION_ENABLED,
            "diversity_cap_enabled":        self.DIVERSITY_CAP_ENABLED,
            "intermittent_routing_enabled": self.INTERMITTENT_ROUTING_ENABLED,
        }

    def __repr__(self) -> str:
        return (
            f"SentinelConfig("
            f"tier={self.ACTIVE_TIER!r}, "
            f"backtest={self.BACKTEST_ENABLED}, "
            f"mase_exclusion={self.MASE_EXCLUSION_ENABLED}, "
            f"diversity_cap={self.DIVERSITY_CAP_ENABLED}, "
            f"intermittent_routing={self.INTERMITTENT_ROUTING_ENABLED})"
        )


# --------------------------------------------------
# MODULE-LEVEL SINGLETON
# --------------------------------------------------

_config = SentinelConfig()


# --------------------------------------------------
# PUBLIC ACCESSORS
# --------------------------------------------------

def get_config() -> SentinelConfig:
    """
    Return the active SentinelConfig singleton.
    This is the primary accessor for all engine components.
    """
    return _config


def set_tier(tier: str) -> None:
    """
    Set the active platform tier.

    Args:
        tier: "essentials" | "pro" | "enterprise"

    Raises:
        ValueError: if tier is not a valid value.

    Effect:
        Filters the model registry to the specified tier.
        Models above the tier will not run.
        Takes effect on the next run_all_models() call.
    """
    if tier not in VALID_TIERS:
        raise ValueError(
            f"Invalid tier: {tier!r}. "
            f"Must be one of: {sorted(VALID_TIERS)}"
        )
    _config.ACTIVE_TIER = tier


def set_flag(flag: str, value: bool) -> None:
    """
    Set a feature flag by name.

    Args:
        flag:  One of VALID_FLAGS
        value: True | False

    Raises:
        ValueError: if flag name is not recognised.
        TypeError:  if value is not bool.
    """
    if flag not in VALID_FLAGS:
        raise ValueError(
            f"Unknown flag: {flag!r}. "
            f"Valid flags: {sorted(VALID_FLAGS)}"
        )
    if not isinstance(value, bool):
        raise TypeError(f"Flag value must be bool, got {type(value).__name__}")
    setattr(_config, flag, value)


def reset_config() -> None:
    """
    Reset all config to production defaults.
    Useful in tests to ensure clean state between runs.
    """
    global _config
    _config = SentinelConfig()


def get_active_tier() -> str:
    """Convenience accessor for the active tier string."""
    return _config.ACTIVE_TIER