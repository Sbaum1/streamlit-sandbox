# 🔒 LOCKED FILE (READ-ONLY AUDIT PANEL)
# ==================================================
# FILE: features/home/scenario_audit_panel.py
# ROLE: SCENARIO AUDIT — EXECUTIVE TRACEABILITY
# STATUS: ADDITIVE / GOVERNANCE-COMPLIANT
#
# PURPOSE:
# - Display scenario metadata for executive audit
# - Provide traceability and transparency
#
# GOVERNANCE:
# - READS st.session_state only
# - NO mutation
# - NO forecasting logic
# - NO side effects
# ==================================================

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_scenario_audit_panel():
    """
    Render an executive audit panel describing the active scenario.

    Requires (read-only):
      - st.session_state.active_scenario_audit (dict)
    """

    audit = st.session_state.get("active_scenario_audit")

    if not isinstance(audit, dict) or not audit:
        return

    st.markdown("### Scenario Audit")

    scenario_name = audit.get("scenario_name", "Unnamed Scenario")
    scenario_type = audit.get("scenario_type", "Unknown")
    timestamp = audit.get("timestamp", "Unknown")
    description = audit.get("scenario_description", "")

    st.caption(
        f"Audit record for **{scenario_name}** "
        f"(type: {scenario_type}, created: {timestamp})"
    )

    # --------------------------------------------------
    # PARAMETER TABLE
    # --------------------------------------------------
    params = {
        k: v
        for k, v in audit.items()
        if k
        not in {
            "scenario_name",
            "scenario_type",
            "scenario_description",
            "timestamp",
        }
    }

    if params:
        df = pd.DataFrame(
            {
                "Parameter": list(params.keys()),
                "Value": list(params.values()),
            }
        )
        st.dataframe(df, use_container_width=True)

    if description:
        st.markdown("**Scenario Description**")
        st.write(description)
