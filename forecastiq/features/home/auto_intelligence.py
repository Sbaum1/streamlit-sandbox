# ==================================================
# FILE: features/home/auto_intelligence.py
# ROLE: AUTO-INTELLIGENCE ADVISOR + TOGGLE (LOCKED)
# STATUS: EXECUTIVE-GRADE / STABLE
#
# GOVERNANCE:
# - Sidebar shows TOGGLE ONLY (NO DATA, NO METRICS)
# - This module MUST NOT trigger forecasts
# - This module MUST ONLY set session_state recommendations
# - All explanatory output MUST render in main content via a separate module
# ==================================================

import streamlit as st
import pandas as pd


# --------------------------------------------------
# INTERNAL: RECOMMENDATION LOGIC (PURE / NO SIDE EFFECTS)
# --------------------------------------------------

def _recommend_settings(df: pd.DataFrame, freq: str) -> dict:
    """
    Generate executive-safe forecast recommendations based on
    data length and inferred frequency.
    """
    n = len(df)

    # Horizon + seasonality heuristics
    if freq == "Monthly":
        horizon = 12
        seasonality = "Yearly"
    elif freq == "Weekly":
        horizon = 26
        seasonality = "Quarterly"
    elif freq == "Daily":
        horizon = 30
        seasonality = "Weekly"
    else:
        horizon = max(12, int(n * 0.25))
        seasonality = "None"

    # Metric & confidence posture
    metric = "RMSE" if n >= 24 else "MAE"
    confidence = 95 if n >= 36 else 90

    return {
        "horizon": horizon,
        "seasonality": seasonality,
        "metric": metric,
        "confidence": confidence,
    }


# --------------------------------------------------
# PUBLIC: SIDEBAR TOGGLE ONLY (NO OUTPUT)
# --------------------------------------------------

def render_auto_intelligence_panel():
    """
    Sidebar-safe Auto-Intelligence toggle.

    This function:
    - Renders ONLY a toggle in the sidebar
    - Writes recommendations silently to session_state
    - NEVER renders metrics, data, or explanations in the sidebar
    """

    with st.sidebar:
        st.markdown("### Auto-Intelligence")

        enabled = st.toggle(
            "Enable Executive Auto-Intelligence",
            value=st.session_state.get("auto_intelligence", False),
            help=(
                "Enables executive-grade forecast recommendations. "
                "Recommendations are advisory only and do not run forecasts."
            ),
        )

    # Persist toggle state
    st.session_state["auto_intelligence"] = enabled

    if not enabled:
        return

    # --- Silent recommendation computation (NO UI) ---
    df = st.session_state.get("committed_df")
    freq = st.session_state.get("data_frequency", "Unknown")

    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    rec = _recommend_settings(df, freq)

    # Write recommendations to session_state ONLY
    st.session_state.update(
        {
            "recommended_forecast_horizon": rec["horizon"],
            "recommended_seasonality_mode": rec["seasonality"],
            "recommended_metric_priority": rec["metric"],
            "recommended_confidence_level": rec["confidence"],
        }
    )
