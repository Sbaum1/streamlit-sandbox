# ==================================================
# FILE: forecastiq/state/session.py
# VERSION: 3.0.0
# ROLE: AUTHORITATIVE SESSION STATE
# UPDATED: Phase 5 — Auto-Intelligence, CI width,
#          ensemble weights, outlier sensitivity,
#          macro variable regressors (Tier A + B)
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
    "backtest_horizon":         12,
    "forecast_horizon":         12,
    "horizon_unit":             None,

    # ── Engine configuration (Phase 5) ───────────
    "ci_level":                 0.95,        # 0.80 / 0.90 / 0.95 / 0.99
    "backtest_strategy":        "expanding", # expanding / rolling
    "ensemble_weight_method":   "mase",      # mase / equal / bayesian
    "outlier_sensitivity":      "medium",    # low / medium / high / none
    "analyst_mode":             False,

    # ── Auto-Intelligence ─────────────────────────
    "auto_intelligence":        False,       # master toggle
    "ai_insights_cache":        {},          # {section_key: insight_text}

    # ── Macro variables — Tier A (FRED regressors) ─
    "macro_vars_enabled":       False,
    "macro_fred_key":           None,        # FRED API key (user supplied)
    "macro_selected_series":    [],          # list of FRED series IDs
    "macro_exog_df":            None,        # DataFrame of fetched macro data
    "macro_exog_future":        None,        # User-supplied forward values

    # ── Macro variables — Tier B (multipliers) ───
    "macro_multipliers":        {},          # {var_name: shock_pct}
    "macro_multiplier_enabled": False,

    # ── Legacy model outputs (UI contract) ───────
    "latest_model_name":        None,
    "latest_model_override":    None,
    "latest_metrics":           None,
    "latest_forecasts":         None,
    "latest_forecast_df":       None,
    "latest_forecast_series":   None,
    "latest_intervals":         None,
    "latest_diagnostics":       None,

    # ── Scenario ─────────────────────────────────
    "scenario_state":           {"enabled": False, "params": None, "applied_signature": None},
    "scenario_forecast_df":     None,

    # ── Figures ──────────────────────────────────
    "latest_fig_inputs":        None,
    "latest_fig_scenario":      None,

    # ── Sentinel Engine outputs ───────────────────
    "sentinel_results":         None,
    "sentinel_engine_version":  None,
    "sentinel_active_tier":     "enterprise",
    "sentinel_primary_df":      None,
    "sentinel_stacked_df":      None,
    "sentinel_cert_metadata":   None,
    "sentinel_run_metadata":    None,

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
