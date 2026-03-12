# 🔒 LOCKED FILE (EXECUTIVE INTENT PANEL)
# ==================================================
# FILE: forecastiq/features/intent/executive_intent_panel.py
# ROLE: Executive Intent Capture (UX + Normalization)
# STATUS: CANONICAL / GOVERNANCE-COMPLIANT
#
# PURPOSE:
# - Capture executive assumptions about macro environment
# - Normalize intent into a reusable schema
#
# GOVERNANCE:
# - UI ONLY
# - Writes st.session_state["executive_intent"]
# - NO forecasting logic
# - NO scenario math
# - NO verdict logic
# ==================================================

from __future__ import annotations

import streamlit as st
from datetime import datetime


# ==================================================
# INTERNAL HELPERS
# ==================================================

def _default_intent() -> dict:
    """
    Canonical default intent state.
    """
    return {
        "macro_regime": "baseline",   # baseline | recession | growth | external_shock
        "severity": "none",           # none | light | moderate | severe
        "duration": "short",          # short | medium | prolonged
        "confidence": "high",          # high | medium | low
        "signals": {
            "credit": "low",
            "employment": "low",
            "demand": "low",
            "rates": "low",
        },
        "risk_posture": "balanced",   # aggressive | balanced | defensive
        "timestamp": None,
        "intent_version": "v1",
    }


def _ensure_intent_state():
    """
    Ensure executive_intent exists in session_state.
    """
    if "executive_intent" not in st.session_state:
        st.session_state["executive_intent"] = _default_intent()


# ==================================================
# MAIN RENDER
# ==================================================

def render_executive_intent_panel():
    """
    Executive Intent Panel

    Collects executive assumptions about:
    - Macro regime
    - Recession severity (conditional)
    - Duration
    - Confidence
    - Economic stress signals
    - Organizational risk posture

    Writes a normalized object to:
      st.session_state["executive_intent"]
    """

    _ensure_intent_state()
    intent = st.session_state["executive_intent"]

    st.markdown("## Executive Outlook & Intent")
    st.caption(
        "Capture your current macro assumptions. These inputs inform scenario stress, "
        "risk posture, and executive decision framing."
    )

    st.divider()

    # --------------------------------------------------
    # SECTION A — MACRO REGIME
    # --------------------------------------------------

    macro_regime = st.radio(
        "Macro environment you are planning for:",
        options=["baseline", "recession", "growth", "external_shock"],
        format_func=lambda x: {
            "baseline": "Baseline / Business as usual",
            "recession": "Recession",
            "growth": "Growth acceleration",
            "external_shock": "External shock",
        }[x],
        index=["baseline", "recession", "growth", "external_shock"].index(
            intent.get("macro_regime", "baseline")
        ),
    )

    intent["macro_regime"] = macro_regime

    # --------------------------------------------------
    # SECTION B — RECESSION SEVERITY (CONDITIONAL)
    # --------------------------------------------------

    severity = "none"
    if macro_regime == "recession":
        severity = st.selectbox(
            "Recession severity:",
            options=["light", "moderate", "severe"],
            index=["light", "moderate", "severe"].index(
                intent.get("severity", "light")
                if intent.get("severity") in ["light", "moderate", "severe"]
                else "light"
            ),
            format_func=lambda x: {
                "light": "Light recession",
                "moderate": "Moderate recession",
                "severe": "Severe recession",
            }[x],
        )

    intent["severity"] = severity

    # --------------------------------------------------
    # SECTION C — DURATION EXPECTATION
    # --------------------------------------------------

    duration = st.selectbox(
        "Expected duration of this environment:",
        options=["short", "medium", "prolonged"],
        index=["short", "medium", "prolonged"].index(
            intent.get("duration", "short")
            if intent.get("duration") in ["short", "medium", "prolonged"]
            else "short"
        ),
        format_func=lambda x: {
            "short": "Short (≤ 2 quarters)",
            "medium": "Medium (2–4 quarters)",
            "prolonged": "Prolonged (4+ quarters)",
        }[x],
    )

    intent["duration"] = duration

    # --------------------------------------------------
    # SECTION D — CONFIDENCE
    # --------------------------------------------------

    confidence = st.selectbox(
        "Confidence in this outlook:",
        options=["high", "medium", "low"],
        index=["high", "medium", "low"].index(
            intent.get("confidence", "high")
            if intent.get("confidence") in ["high", "medium", "low"]
            else "high"
        ),
        format_func=lambda x: {
            "high": "High confidence",
            "medium": "Moderate confidence",
            "low": "Low confidence",
        }[x],
    )

    intent["confidence"] = confidence

    # --------------------------------------------------
    # SECTION E — ECONOMIC SIGNAL STRESS
    # --------------------------------------------------

    st.markdown("### Economic Stress Signals")
    st.caption("Qualitative signals currently observed.")

    signals = intent.get("signals", {})

    def _signal_selector(label: str, key: str):
        return st.radio(
            label,
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(
                signals.get(key, "low")
                if signals.get(key) in ["low", "medium", "high"]
                else "low"
            ),
            horizontal=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        signals["credit"] = _signal_selector("Credit conditions", "credit")
        signals["employment"] = _signal_selector("Employment", "employment")
    with c2:
        signals["demand"] = _signal_selector("Demand", "demand")
        signals["rates"] = _signal_selector("Interest rates", "rates")

    intent["signals"] = signals

    # --------------------------------------------------
    # SECTION F — RISK POSTURE
    # --------------------------------------------------

    risk_posture = st.selectbox(
        "Organizational risk posture:",
        options=["aggressive", "balanced", "defensive"],
        index=["aggressive", "balanced", "defensive"].index(
            intent.get("risk_posture", "balanced")
            if intent.get("risk_posture") in ["aggressive", "balanced", "defensive"]
            else "balanced"
        ),
        format_func=lambda x: {
            "aggressive": "Aggressive (lean in)",
            "balanced": "Balanced",
            "defensive": "Defensive (preserve capital)",
        }[x],
    )

    intent["risk_posture"] = risk_posture

    # --------------------------------------------------
    # FINALIZE & COMMIT
    # --------------------------------------------------

    intent["timestamp"] = datetime.utcnow().isoformat()
    st.session_state["executive_intent"] = intent

    st.success("Executive intent captured and applied.")
