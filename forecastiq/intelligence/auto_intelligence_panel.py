# 🔒 LOCKED FILE (AUTO-INTELLIGENCE EXPLANATION PANEL)
# ==================================================
# FILE: forecastiq/intelligence/auto_intelligence_panel.py
# ROLE: Executive Auto-Intelligence Narrative & Framing
# STATUS: CANONICAL / GOVERNANCE-COMPLIANT / EXECUTIVE-GRADE
#
# PURPOSE:
# - Explain WHY the executive verdict looks the way it does
# - Surface intent-driven risk framing (blue banner)
# - Translate math + intent into executive language
#
# GOVERNANCE:
# - READS st.session_state only
# - NO forecasting logic
# - NO verdict mutation
# - NO scenario mutation
# - NO sidebar rendering
# - UI + narrative ONLY
# ==================================================

from __future__ import annotations

import streamlit as st


# ==================================================
# INTERNAL HELPERS (PURE)
# ==================================================

def _intent_banner(intent: dict) -> tuple[str, str] | None:
    """
    Returns (banner_text, banner_type) or None
    banner_type ∈ {"info", "warning", "error"}
    """

    macro = intent.get("macro_regime", "baseline")
    severity = intent.get("severity", "none")
    confidence = intent.get("confidence", "high")
    risk_posture = intent.get("risk_posture", "balanced")

    if macro != "recession":
        return None

    if severity == "light":
        return (
            "You are planning under a **Light Recession** assumption. "
            "Upside optimism is intentionally capped to surface early-warning risk, "
            "even if near-term performance remains strong.",
            "info",
        )

    if severity == "moderate":
        return (
            "You are planning under a **Moderate Recession** assumption. "
            "This enforces a cautionary posture and elevates downside risk visibility "
            "across forecasts and decisions.",
            "warning",
        )

    if severity == "severe":
        return (
            "You are planning under a **Severe Recession** assumption. "
            "Risk tolerance is constrained and defensive bias is applied to prevent "
            "over-commitment under stress.",
            "error",
        )

    return None


def _signal_summary(signals: dict) -> list[str]:
    """
    Translate qualitative stress signals into readable bullets.
    """
    bullets = []

    for key, level in signals.items():
        if level == "high":
            bullets.append(f"🔺 Elevated stress detected in **{key.capitalize()}**")
        elif level == "medium":
            bullets.append(f"🟡 Moderate stress detected in **{key.capitalize()}**")

    return bullets


def _verdict_explanation(verdict: str, intent: dict) -> str:
    """
    Plain-English explanation of verdict framing.
    """

    macro = intent.get("macro_regime", "baseline")
    severity = intent.get("severity", "none")

    if macro == "recession" and severity == "light" and verdict.startswith("GO"):
        return (
            "Although the forecast remains directionally positive, the decision has "
            "been framed conservatively to reflect your recession assumption. "
            "This ensures early signals are not masked by short-term strength."
        )

    if macro == "recession" and severity == "moderate":
        return (
            "The verdict reflects a balance between forecast performance and "
            "heightened macro uncertainty. Selective execution is advised."
        )

    if macro == "recession" and severity == "severe":
        return (
            "Decision framing prioritizes capital preservation and downside control "
            "given the severity of the assumed macro stress."
        )

    return (
        "The decision reflects forecast performance under current assumptions. "
        "No additional macro risk framing is applied."
    )


# ==================================================
# PUBLIC RENDER FUNCTION
# ==================================================

def render_auto_intelligence_panel():
    """
    Render executive-grade Auto-Intelligence explanation panel.

    This panel:
    - Explains executive intent impact
    - Surfaces macro framing (blue banner)
    - Clarifies why verdict may appear conservative
    """

    # Feature toggle gate
    if not st.session_state.get("auto_intelligence", False):
        return

    intent = st.session_state.get("executive_intent", {})
    verdict_text = st.session_state.get("latest_verdict_text")

    st.markdown("## Executive Auto-Intelligence")

    # --------------------------------------------------
    # INTENT BANNER (BLUE / YELLOW / RED)
    # --------------------------------------------------

    banner = _intent_banner(intent)
    if banner:
        text, level = banner
        if level == "info":
            st.info(text)
        elif level == "warning":
            st.warning(text)
        else:
            st.error(text)

    # --------------------------------------------------
    # ECONOMIC SIGNAL SUMMARY
    # --------------------------------------------------

    signals = intent.get("signals", {})
    signal_bullets = _signal_summary(signals)

    if signal_bullets:
        st.markdown("### Observed Economic Signals")
        for b in signal_bullets:
            st.markdown(f"- {b}")

    # --------------------------------------------------
    # VERDICT FRAMING
    # --------------------------------------------------

    if verdict_text:
        st.markdown("### Why This Decision Looks This Way")
        explanation = _verdict_explanation(verdict_text, intent)
        st.caption(explanation)

    st.divider()
    st.caption(
        "Auto-Intelligence provides interpretive guidance only. "
        "All forecasts and verdicts remain mathematically grounded and auditable."
    )
