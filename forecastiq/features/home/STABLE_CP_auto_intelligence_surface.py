# ==================================================
# FILE: features/home/auto_intelligence_surface.py
# ROLE: AUTO-INTELLIGENCE EXPLANATION SURFACE (READ-ONLY)
# STATUS: EXECUTIVE-GRADE / GOVERNANCE-COMPLIANT
#
# PURPOSE:
# - Render WHY the verdict occurred
# - Explain math ↔ scenario ↔ intent interaction
# - Surface early warnings and conflicts
#
# GOVERNANCE:
# - READS st.session_state only
# - NO forecasting logic
# - NO verdict mutation
# - NO scenario mutation
# ==================================================

import streamlit as st

from forecastiq.intelligence.auto_intelligence_engine import (
    generate_intelligence,
    IntelligenceSignal,
)


# ==================================================
# SEVERITY → VISUAL MAPPING
# ==================================================

_SEVERITY_STYLES = {
    "info": ("ℹ️", st.info),
    "early": ("🔵", st.info),
    "moderate": ("🟡", st.warning),
    "high": ("🟠", st.warning),
    "critical": ("🔴", st.error),
}


# ==================================================
# MAIN RENDER
# ==================================================

def render_auto_intelligence_surface():
    """
    Render Auto-Intelligence explanations for the active run.

    This surface explains:
    - Why the verdict was reached
    - How executive intent influenced outcomes
    - Where early risk signals exist
    """

    if not st.session_state.get("auto_intelligence", False):
        return

    baseline_df = st.session_state.get("latest_forecast_df")
    scenario_df = st.session_state.get("scenario_forecast_df")
    intent = st.session_state.get("executive_intent", {})
    audit = st.session_state.get("active_scenario_audit", {})

    verdict_payload = st.session_state.get("executive_verdict_payload")

    if baseline_df is None or verdict_payload is None:
        return

    signals = generate_intelligence(
        baseline_df=baseline_df,
        scenario_df=scenario_df,
        verdict_payload=verdict_payload,
        executive_intent=intent,
        scenario_audit=audit,
    )

    if not signals:
        return

    st.markdown("## 🧠 Executive Auto-Intelligence")
    st.caption(
        "Interpretive guidance explaining how forecast math, scenario stress, "
        "and executive intent combined to produce the final verdict."
    )

    for sig in signals:
        _render_signal(sig)


# ==================================================
# SIGNAL RENDERER
# ==================================================

def _render_signal(signal: IntelligenceSignal):
    icon, renderer = _SEVERITY_STYLES.get(
        signal.severity, ("ℹ️", st.info)
    )

    content = f"""
**{icon} {signal.title}**

{signal.message}

**Recommendation:**  
{signal.recommendation}

**Confidence:** {int(signal.confidence * 100)}%
"""

    renderer(content)

    if signal.follow_up:
        with st.expander("Details"):
            st.json(signal.follow_up)
