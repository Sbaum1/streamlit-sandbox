# ==================================================
# FILE: forecastiq/intelligence/auto_intelligence_engine.py
# ROLE: Auto-Intelligence Reasoning Engine (PURE LOGIC)
# STATUS: CANONICAL / EXECUTIVE-GRADE / EXPANDABLE
#
# PURPOSE:
# - Generate structured executive intelligence signals
# - Explain WHY outcomes occurred (not WHAT to do)
# - Bridge math, scenario stress, and executive intent
#
# GOVERNANCE:
# - NO Streamlit imports
# - NO session_state access
# - NO forecasting logic
# - NO verdict mutation
# - PURE functions only
#
# DESIGN PRINCIPLES:
# - Deterministic
# - Auditable
# - Expandable
# - Reusable across all tabs
# ==================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# ==================================================
# INTELLIGENCE DATA STRUCTURE
# ==================================================

@dataclass(frozen=True)
class IntelligenceSignal:
    """
    Canonical intelligence signal returned to UI layer.

    severity:
      - info        : explanatory only
      - early       : early warning
      - moderate    : caution warranted
      - high        : material concern
      - critical    : executive attention required
    """

    id: str
    type: str
    severity: str
    title: str
    message: str
    recommendation: str
    confidence: float
    follow_up: Optional[Dict[str, Any]] = None


# ==================================================
# PUBLIC API
# ==================================================

def generate_intelligence(
    *,
    baseline_df,
    scenario_df,
    verdict_payload: Dict[str, Any],
    executive_intent: Dict[str, Any],
    scenario_audit: Dict[str, Any],
) -> List[IntelligenceSignal]:
    """
    Generate executive intelligence signals based on:
    - Baseline vs scenario math
    - Final verdict outcome
    - Declared executive intent
    - Scenario metadata

    Returns a list of IntelligenceSignal objects.
    """

    signals: List[IntelligenceSignal] = []

    # --------------------------------------------------
    # Guardrails
    # --------------------------------------------------
    if baseline_df is None or verdict_payload is None:
        return signals

    # Extract commonly used fields safely
    verdict = verdict_payload.get("verdict")
    downside_pct = verdict_payload.get("downside_pct", 0.0)

    macro = executive_intent.get("macro_regime", "baseline")
    severity = executive_intent.get("severity", "none")
    confidence = executive_intent.get("confidence", "medium")
    risk_posture = executive_intent.get("risk_posture", "balanced")

    scenario_name = scenario_audit.get("scenario_name", "Baseline")

    # ==================================================
    # DETECTOR 1 — INTENT ↔ MATH CONFLICT
    # ==================================================
    if macro == "recession" and downside_pct > -4 and verdict != "GO":
        signals.append(
            IntelligenceSignal(
                id="intent_math_conflict",
                type="intent_conflict",
                severity="early",
                title="Executive Intent Overrides Mathematical Outlook",
                message=(
                    "The selected recession scenario introduces caution despite "
                    "limited mathematical downside in the forecast."
                ),
                recommendation=(
                    "This outcome reflects executive risk framing rather than forecast stress. "
                    "Consider validating recession assumptions against incoming data."
                ),
                confidence=0.75,
                follow_up={
                    "macro_regime": macro,
                    "declared_severity": severity,
                    "measured_downside_pct": round(downside_pct, 2),
                },
            )
        )

    # ==================================================
    # DETECTOR 2 — SEVERITY CAP EXPLANATION
    # ==================================================
    if macro == "recession" and severity in ("light", "moderate"):
        if verdict in ("GO — MONITOR", "PROCEED WITH CAUTION"):
            signals.append(
                IntelligenceSignal(
                    id="severity_verdict_cap",
                    type="verdict_cap",
                    severity="info",
                    title="Verdict Capped by Recession Severity Selection",
                    message=(
                        f"A {severity} recession was selected, which limits upside verdicts "
                        "even when forecast performance remains resilient."
                    ),
                    recommendation=(
                        "If confidence in recession severity decreases, re-run scenarios "
                        "with adjusted assumptions to reassess upside potential."
                    ),
                    confidence=0.85,
                    follow_up={
                        "scenario": scenario_name,
                        "severity": severity,
                        "risk_posture": risk_posture,
                    },
                )
            )

    # ==================================================
    # DETECTOR 3 — EARLY DOWNSIDE SHAPE WARNING
    # ==================================================
    if scenario_df is not None:
        try:
            future = scenario_df[scenario_df["is_future"] == True]["forecast"]
            if len(future) >= 3:
                early_change = future.iloc[1] - future.iloc[0]
                total_change = future.iloc[-1] - future.iloc[0]

                if early_change < 0 and total_change > early_change:
                    signals.append(
                        IntelligenceSignal(
                            id="front_loaded_downside",
                            type="downside_shape",
                            severity="early",
                            title="Front-Loaded Downside Detected",
                            message=(
                                "The scenario shows early-period weakness followed by stabilization. "
                                "Short-term risk may be understated in aggregate metrics."
                            ),
                            recommendation=(
                                "Monitor near-term indicators closely and prepare short-horizon mitigations."
                            ),
                            confidence=0.7,
                            follow_up={
                                "early_change": float(early_change),
                                "net_change": float(total_change),
                            },
                        )
                    )
        except Exception:
            pass  # Fail-safe: intelligence must never break execution

    # ==================================================
    # DETECTOR 4 — DEFENSIVE POSTURE AMPLIFICATION
    # ==================================================
    if risk_posture == "defensive" and verdict in ("GO", "GO — MONITOR"):
        signals.append(
            IntelligenceSignal(
                id="defensive_bias_active",
                type="posture_bias",
                severity="info",
                title="Defensive Risk Posture Applied",
                message=(
                    "A defensive organizational posture has reduced tolerance for upside risk "
                    "despite forecast stability."
                ),
                recommendation=(
                    "This is consistent with capital preservation strategies. "
                    "No action required unless posture changes."
                ),
                confidence=0.9,
            )
        )

    return signals
