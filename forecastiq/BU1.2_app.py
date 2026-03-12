# ==================================================
# FILE: veduta/app.py
# ROLE: ROUTER ONLY (LOCKED)
# UPDATED: VEDUTA Rebrand v1.0
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
    page_title="VEDUTA | Executive Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ── Global VEDUTA styles — loaded once, apply to every tab ──────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #07080F; }
[data-testid="stSidebar"]          { background: #0D1420; border-right: 1px solid #243347; }
.block-container { padding-top: 1rem !important; max-width: 1400px; }

/* ── Tab bar — always visible ── */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: #4A6278 !important;
    background: transparent !important;
    border: none !important;
    padding: .65rem 1.25rem !important;
}
button[data-baseweb="tab"]:hover {
    color: #8FA3B8 !important;
    background: rgba(200,151,74,.07) !important;
    border-radius: 4px 4px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #EDE8DE !important;
    border-bottom: 2px solid #C8974A !important;
}
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #243347 !important;
    gap: 0 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #C8974A !important;
    height: 2px !important;
}
div[data-baseweb="tab-border"] {
    background-color: #243347 !important;
    height: 1px !important;
}

/* ── Global typography ── */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    color: #EDE8DE !important;
    letter-spacing: 0.04em !important;
}
p, li, label {
    font-family: 'DM Mono', monospace !important;
    color: #8FA3B8 !important;
}
hr { border-color: #243347 !important; }

/* ── Metric widgets ── */
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: .6rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: #4A6278 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
    font-weight: 300 !important;
    color: #EDE8DE !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    background: #1B2A40 !important;
    border: 1px solid #243347 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .72rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1B2A40 !important;
    border: 1px solid #243347 !important;
    border-radius: 10px !important;
}

/* ── File upload ── */
[data-testid="stFileUploadDropzone"] {
    background: #1B2A40 !important;
    border: 1.5px dashed #243347 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: #C8974A !important; }

/* ── Sidebar collapse button fix ── */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * {
    font-family: 'Material Icons' !important;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

render_command_center()

tabs = st.tabs([
    "The Veduta",
    "Intelligence & Trust",
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
