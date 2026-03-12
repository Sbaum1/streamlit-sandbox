# ==================================================
# FILE: forecastiq/state/session.py
# VERSION: 2.0.0
# ROLE: AUTHORITATIVE SESSION STATE
# UPDATED: Phase 4 — Sentinel Engine session keys added
# ==================================================
import streamlit as st
from datetime import datetime

REQUIRED_KEYS = {
    # ── Core data ────────────────────────────────
    "committed_df":             None,
    "committed_meta":           None,
    "data_frequency":           None,
    "freq_inference_details":   None,

    # ── Run state ────────────────────────────────
    "forecast_status":          "Idle",
    "last_run_timestamp":       None,
    "run_signature":            None,
    "backtest_horizon":         None,
    "forecast_horizon":         None,
    "horizon_unit":             None,

    # ── Legacy model outputs (UI contract) ───────
    "latest_model_name":        None,
    "latest_model_override":    None,
    "latest_metrics":           None,
    "latest_forecasts":         None,
    "latest_forecast_df":       None,
    "latest_forecast_series":   None,   # Series alias for executive_summary / report_builder
    "latest_intervals":         None,
    "latest_diagnostics":       None,

    # ── Scenario ─────────────────────────────────
    "scenario_state":           {"enabled": False, "params": None, "applied_signature": None},
    "scenario_forecast_df":     None,

    # ── Figures ──────────────────────────────────
    "latest_fig_inputs":        None,
    "latest_fig_scenario":      None,

    # ── Sentinel Engine outputs (Phase 4) ────────
    "sentinel_results":         None,   # full results dict from run_all_models()
    "sentinel_engine_version":  None,   # ENGINE_VERSION string
    "sentinel_active_tier":     "enterprise",  # enterprise / pro / essentials
    "sentinel_primary_df":      None,   # Primary Ensemble forecast_df
    "sentinel_stacked_df":      None,   # Stacked Ensemble forecast_df
    "sentinel_cert_metadata":   None,   # per-model MASE, tier, CI method
    "sentinel_run_metadata":    None,   # _engine block from results

    # ── Audit ─────────────────────────────────────
    "audit_log":                [],
}


def init_state():
    for key, default in REQUIRED_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if not st.session_state.audit_log:
        st.session_state.audit_log.append({
            "event":     "app_init",
            "timestamp": datetime.utcnow().isoformat(),
            "summary":   "Application session initialized",
        })
