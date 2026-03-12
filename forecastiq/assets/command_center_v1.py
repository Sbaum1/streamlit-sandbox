# ==================================================
# FILE: forecastiq/sidebar/command_center.py
# ROLE: EXECUTIVE COMMAND CENTER (CONTROLS ONLY)
# STATUS: LOCKED / EXECUTIVE-GRADE
#
# GOVERNANCE:
# - Sidebar MUST contain controls and static badges only
# - NO data output, JSON, expanders, or runtime messages
# - Execution feedback handled outside sidebar
# ==================================================

from __future__ import annotations

import streamlit as st
from datetime import datetime, timezone

from engine.forecast import run_all_models


# --------------------------------------------------
# STATE INITIALIZATION (DEFENSIVE, NON-DESTRUCTIVE)
# --------------------------------------------------

def _ensure_state_keys():
    defaults = {
        "committed_df": None,
        "committed_meta": {},
        "data_frequency": "Unknown",
        "freq_inference_details": {},
        "forecast_status": "Idle",
        "last_run_timestamp": None,
        "run_signature": None,
        "backtest_horizon": 6,
        "forecast_horizon": 6,
        "horizon_unit": "months",
        "confidence_level": 95,
        "uncertainty_mode": "Neutral",
        "metric_priority": "RMSE",
        "business_objective": "Balanced",
        "forecast_intent": "Planning",
        "model_selection_strategy": "auto",
        "run_lock": False,
        "explain_forecast": False,
        "latest_model_name": None,
        "latest_model_override": None,
        "latest_metrics": None,
        "latest_forecasts": {},
        "latest_forecast_df": None,
        "latest_intervals": {},
        "latest_diagnostics": {},
        "audit_log": [],
        "analyst_mode": False,

        # 🔹 NEW — RUN RESOLUTION (A/B)
        "run_resolution_mode": "auto",     # auto | forced
        "forced_model": None,               # arima | ets | prophet | naive

        # Advanced parameters (authoritative)
        "advanced_params": {
            "prophet_growth": "auto",
            "prophet_seasonality_mode": "additive",
            "prophet_changepoint_prior": 0.10,
            "prophet_yearly": True,
            "prophet_weekly": True,
            "enable_outlier_cleaning": False,
            "outlier_z_threshold": 3.0,
        },
    }

    for k, v in defaults.items():
        if k not in st.session_state or st.session_state[k] is None:
            st.session_state[k] = v

    if not isinstance(st.session_state["advanced_params"], dict):
        st.session_state["advanced_params"] = defaults["advanced_params"].copy()


# --------------------------------------------------
# SMALL UI HELPERS (TEXT ONLY)
# --------------------------------------------------

def _status_badge(label: str, value: str):
    st.markdown(f"**{label}:** {value}")


def _data_status() -> str:
    return "Committed" if st.session_state.get("committed_df") is not None else "Not committed"


def _safe_index(options, value, fallback=0):
    try:
        return options.index(value)
    except Exception:
        return fallback


def _safe_clear_computed_artifacts():
    st.session_state.update(
        {
            "forecast_status": "Idle",
            "last_run_timestamp": None,
            "run_signature": None,
            "latest_model_name": None,
            "latest_model_override": None,
            "latest_metrics": None,
            "latest_forecasts": {},
            "latest_forecast_df": None,
            "latest_intervals": {},
            "latest_diagnostics": {},
        }
    )


# --------------------------------------------------
# MAIN SIDEBAR RENDER (CONTROLS ONLY)
# --------------------------------------------------

def render_command_center():
    _ensure_state_keys()

    with st.sidebar:
        st.markdown("## Executive Command Center")

        # ==================================================
        # STATUS
        # ==================================================
        st.markdown("### Status")

        _status_badge("Forecast Status", st.session_state["forecast_status"])
        _status_badge("Data Status", _data_status())

        freq = st.session_state["data_frequency"]
        conf = st.session_state["freq_inference_details"].get("confidence", "—")
        _status_badge("Frequency", f"{freq} (conf: {conf})")

        _status_badge("Last Run", st.session_state["last_run_timestamp"] or "—")

        st.divider()

        # ==================================================
        # FORECAST CONTROLS
        # ==================================================
        st.markdown("### Forecast Controls")

        with st.form("baseline_forecast_form", clear_on_submit=False):

            # MODEL STRATEGY (AUTO / ETS / PROPHET / BLEND)
            strategy_opts = {
                "Auto (Best Model)": "auto",
                "Prophet": "prophet",
                "ETS": "ets",
                "Blend": "blend",
            }

            strategy_label = st.selectbox(
                "Model Strategy",
                list(strategy_opts.keys()),
                index=_safe_index(
                    list(strategy_opts.values()),
                    st.session_state["model_selection_strategy"],
                ),
            )

            bt = st.number_input("Backtest horizon (periods)", 1, 120, int(st.session_state["backtest_horizon"]))
            fh = st.number_input("Forecast horizon (periods)", 1, 240, int(st.session_state["forecast_horizon"]))

            unit = st.selectbox(
                "Horizon unit",
                ["periods", "weeks", "months"],
                index=_safe_index(["periods", "weeks", "months"], st.session_state["horizon_unit"], 2),
            )

            metric_priority = st.selectbox(
                "Metric priority",
                ["RMSE", "MAE", "MAPE"],
                index=_safe_index(["RMSE", "MAE", "MAPE"], st.session_state["metric_priority"]),
            )

            confidence_level = st.selectbox(
                "Confidence level",
                [80, 90, 95, 99],
                index=_safe_index([80, 90, 95, 99], st.session_state["confidence_level"], 2),
            )

            uncertainty_mode = st.selectbox(
                "Uncertainty posture",
                ["Conservative", "Neutral", "Aggressive"],
                index=_safe_index(["Conservative", "Neutral", "Aggressive"], st.session_state["uncertainty_mode"], 1),
            )

            business_objective = st.selectbox(
                "Business objective",
                ["Growth", "Risk Reduction", "Cost Control", "Balanced"],
                index=_safe_index(["Growth", "Risk Reduction", "Cost Control", "Balanced"], st.session_state["business_objective"], 3),
            )

            forecast_intent = st.selectbox(
                "Forecast usage",
                ["Planning", "Inventory", "Operations", "Board"],
                index=_safe_index(["Planning", "Inventory", "Operations", "Board"], st.session_state["forecast_intent"]),
            )

            run_lock = st.checkbox("Lock forecast after run", value=st.session_state["run_lock"])
            explain = st.checkbox("Explain this forecast", value=st.session_state["explain_forecast"])

            # ==================================================
            # 🔽 ADVANCED RUN RESOLUTION (A/B CONTROLS)
            # ==================================================
            with st.expander("Advanced Run Resolution (A/B Controls)", expanded=False):

                resolution_mode = st.radio(
                    "Run resolution mode",
                    ["Auto (Best Model)", "Force Selected Model"],
                    index=0 if st.session_state["run_resolution_mode"] == "auto" else 1,
                )

                forced_model = None
                if resolution_mode == "Force Selected Model":
                    forced_model = st.selectbox(
                        "Force model for this run",
                        ["ARIMA", "ETS", "Prophet", "Naive"],
                        index=_safe_index(
                            ["ARIMA", "ETS", "Prophet", "Naive"],
                            (st.session_state["forced_model"] or "").upper(),
                        ),
                    )

            # ==================================================
            # PROPHET ADVANCED SETTINGS (PRESERVED)
            # ==================================================
            st.markdown("#### Prophet Advanced Settings")

            ap = st.session_state["advanced_params"]

            growth = st.selectbox(
                "Growth type",
                ["auto", "linear", "logistic"],
                index=_safe_index(["auto", "linear", "logistic"], ap.get("prophet_growth", "auto")),
            )

            seasonality_mode = st.selectbox(
                "Seasonality mode",
                ["additive", "multiplicative"],
                index=_safe_index(["additive", "multiplicative"], ap.get("prophet_seasonality_mode", "additive")),
            )

            cp = st.slider(
                "Changepoint sensitivity",
                0.01,
                0.50,
                float(ap.get("prophet_changepoint_prior", 0.10)),
                step=0.01,
            )

            yearly = st.checkbox("Enable yearly seasonality", value=bool(ap.get("prophet_yearly", True)))
            weekly = st.checkbox("Enable weekly seasonality (if applicable)", value=bool(ap.get("prophet_weekly", True)))

            # ==================================================
            # OUTLIERS & CLEANING (PRESERVED)
            # ==================================================
            st.markdown("#### Outliers & Cleaning")

            enable_outliers = st.checkbox(
                "Enable automatic outlier cleaning",
                value=bool(ap.get("enable_outlier_cleaning", False)),
            )

            z_thresh = st.slider(
                "Outlier z-score threshold",
                2.0,
                5.0,
                float(ap.get("outlier_z_threshold", 3.0)),
                step=0.1,
            )

            col_a, col_b = st.columns([2, 1])
            run_clicked = col_a.form_submit_button("RUN FORECAST")
            reset_clicked = col_b.form_submit_button("RESET")

        # --------------------------------------------------
        # PERSIST CONTROLS
        # --------------------------------------------------
        st.session_state.update(
            {
                "model_selection_strategy": strategy_opts[strategy_label],
                "backtest_horizon": bt,
                "forecast_horizon": fh,
                "horizon_unit": unit,
                "metric_priority": metric_priority,
                "confidence_level": confidence_level,
                "uncertainty_mode": uncertainty_mode,
                "business_objective": business_objective,
                "forecast_intent": forecast_intent,
                "run_lock": run_lock,
                "explain_forecast": explain,

                # 🔹 NEW — RUN RESOLUTION STATE
                "run_resolution_mode": "auto" if resolution_mode == "Auto (Best Model)" else "forced",
                "forced_model": forced_model.lower() if forced_model else None,

                "advanced_params": {
                    "prophet_growth": growth,
                    "prophet_seasonality_mode": seasonality_mode,
                    "prophet_changepoint_prior": cp,
                    "prophet_yearly": yearly,
                    "prophet_weekly": weekly,
                    "enable_outlier_cleaning": enable_outliers,
                    "outlier_z_threshold": z_thresh,
                },
            }
        )

        # --------------------------------------------------
        # EXECUTION (STATE ONLY)
        # --------------------------------------------------
        if reset_clicked:
            _safe_clear_computed_artifacts()

        if run_clicked:
            if not st.session_state["run_lock"] or st.session_state["forecast_status"] != "Complete":
                if st.session_state["committed_df"] is not None:
                    st.session_state["forecast_status"] = "Running"
                    try:
                        out = run_all_models(
                            committed_df=st.session_state["committed_df"],
                            frequency=st.session_state["data_frequency"],
                            backtest_horizon=bt,
                            forecast_horizon=fh,
                            horizon_unit=unit,
                            selection_strategy=st.session_state["model_selection_strategy"],
                            model_override=st.session_state["forced_model"]
                            if st.session_state["run_resolution_mode"] == "forced"
                            else None,
                            advanced_params=st.session_state["advanced_params"],
                        )

                        for k, v in out.items():
                            st.session_state[k] = v

                        st.session_state["forecast_status"] = "Complete"
                        st.session_state["last_run_timestamp"] = datetime.now(timezone.utc).isoformat()

                    except Exception:
                        st.session_state["forecast_status"] = "Failed"
