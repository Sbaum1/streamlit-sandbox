# ==================================================
# FILE: forecastiq/app.py
# ROLE: ROUTER ONLY (LOCKED)
# UPDATED: Phase 5 — Help tab added
# ==================================================
import sys
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import streamlit as st
from state.session import init_state
from sidebar.command_center import render_command_center
from tabs.home import render_home
from tabs.executive_insight_trust import render_executive_insight_trust
from tabs.report_builder import render_report_builder
from tabs.help import render_help

# --------------------------------------------------
# EXECUTIVE BRANDING
# --------------------------------------------------
st.set_page_config(
    page_title="ForecastIQ | Executive Decision Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
render_command_center()

tabs = st.tabs([
    "Decision Center",
    "Executive Insight & Trust",
    "Report Builder",
    "Help & Reference",
])

with tabs[0]:
    render_home()
with tabs[1]:
    render_executive_insight_trust()
with tabs[2]:
    render_report_builder()
with tabs[3]:
    render_help()
