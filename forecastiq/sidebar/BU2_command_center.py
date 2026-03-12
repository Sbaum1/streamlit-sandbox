# ==================================================
# FILE: forecastiq/sidebar/command_center.py
# VERSION: 4.0.0
# ROLE: ELITE EXECUTIVE COMMAND CENTER
# Phase 5: CI width, backtest strategy, ensemble
#          weights, outlier sensitivity, Auto-
#          Intelligence, Tier A FRED regressors,
#          Tier B macro multipliers.
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from engine.forecast import run_forecast
from engine.scenarios import apply_scenario
from engine.macro_variables import (
    FRED_CATALOGUE, TIER_B_PRESETS, get_categories,
    build_exog_df, apply_tier_b_multipliers,
)

_CSS = """
<style>
[data-testid="stSidebar"] { background:#0D1420!important; border-right:1px solid #243347!important; }
[data-testid="stSidebar"] * { font-family:'DM Mono',monospace!important; }

.sb-zone {
    font-size:.58rem; font-weight:700; letter-spacing:.2em;
    text-transform:uppercase; color:#C8974A;
    margin:1.4rem 0 .5rem 0;
    padding-bottom:.35rem; border-bottom:1px solid #243347;
    display:flex; align-items:center; gap:.4rem;
}
.sb-zone .zone-icon { font-size:.7rem; }

.sb-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:.3rem 0; border-bottom:1px solid #0f1929;
}
.sb-key { font-size:.6rem; color:#4b5e80; letter-spacing:.06em; }
.sb-val { font-size:.68rem; font-weight:600; color:#cbd5e1; }
.sb-val.green { color:#6BAF85; }
.sb-val.blue  { color:#C8974A; }
.sb-val.amber { color:#D4834A; }
.sb-val.red   { color:#C45858; }

.ai-badge {
    background:linear-gradient(135deg,rgba(200,151,74,.12),rgba(200,151,74,.06));
    border:1px solid rgba(200,151,74,.3);
    border-radius:8px; padding:.5rem .75rem;
    font-size:.62rem; color:#E2B96A;
    text-align:center; margin:.5rem 0;
    letter-spacing:.04em;
}
.macro-badge {
    background:rgba(34,197,94,.08);
    border:1px solid rgba(34,197,94,.2);
    border-radius:6px; padding:.4rem .75rem;
    font-size:.6rem; color:#86efac;
    margin:.35rem 0; letter-spacing:.04em;
}
.scenario-pill {
    background:rgba(220,38,38,.1); border:1px solid rgba(220,38,38,.3);
    border-radius:6px; padding:.4rem .75rem;
    font-size:.62rem; color:#fca5a5;
    text-align:center; margin:.5rem 0;
}
.run-info {
    background:#07080F; border:1px solid #243347;
    border-radius:8px; padding:.75rem 1rem;
    margin:.5rem 0;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
"""


def _row(k, v, cls=""):
    return (f'<div class="sb-row">'
            f'<span class="sb-key">{k}</span>'
            f'<span class="sb-val {cls}">{v}</span></div>')


def render_command_center():
    st.sidebar.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:

        # ── Brand ────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:.6rem 0 .4rem;border-bottom:1px solid #243347;margin-bottom:.2rem">
          <div style="font-size:.55rem;letter-spacing:.22em;text-transform:uppercase;
                      color:#C8974A;margin-bottom:.15rem">VEDUTA · Foresight Engine</div>
          <div style="font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:300;
                      letter-spacing:.2em;color:#EDE8DE;text-transform:uppercase">Veduta Overview</div>
        </div>
        """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        # ZONE 1 — SYSTEM STATUS
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sb-zone"><span class="zone-icon">◉</span>System Status</div>',
                    unsafe_allow_html=True)

        fc  = st.session_state.forecast_status
        fc_cls = "green" if fc=="Complete" else "amber" if fc=="Running" else "red" if fc=="Failed" else ""
        meta = st.session_state.committed_meta or {}
        committed = st.session_state.committed_df is not None

        data_str = (
            f"{meta.get('rows',0):,}r · "
            f"{meta.get('frequency','?')} · "
            f"{meta['start_date'].strftime('%b %y') if meta.get('start_date') else '?'}"
            f"–{meta['end_date'].strftime('%b %y') if meta.get('end_date') else '?'}"
        ) if committed else "Not Committed"

        last = st.session_state.last_run_timestamp
        last_str = datetime.fromisoformat(last).strftime("%H:%M:%S") if last else "—"

        html = (
            _row("Forecast",  fc,     fc_cls)
            + _row("Dataset", data_str, "green" if committed else "red")
            + _row("Last Run", last_str, "blue" if last else "")
            + _row("Engine",  f"Foresight v{st.session_state.sentinel_engine_version or '1.0'}", "blue")
            + _row("Tier",    (st.session_state.sentinel_active_tier or "—").title(), "blue")
            + _row("CI Level", f"{int(st.session_state.ci_level*100)}%", "")
        )

        # Auto-Intelligence badge
        if st.session_state.auto_intelligence:
            html += '<div class="ai-badge">⚡ Auto-Intelligence Active</div>'

        # Scenario badge
        if st.session_state.scenario_state.get("enabled"):
            p = st.session_state.scenario_state.get("params", {})
            html += (f'<div class="scenario-pill">▲ Scenario · '
                     f'{p.get("type","?")} · {p.get("shock_pct",0):+.1f}%</div>')

        # Macro badge
        if st.session_state.macro_vars_enabled or st.session_state.macro_multiplier_enabled:
            active_count = len(st.session_state.macro_selected_series)
            mult_count   = sum(1 for v in st.session_state.macro_multipliers.values() if v != 0)
            html += (f'<div class="macro-badge">◈ Macro Active · '
                     f'{active_count} FRED · {mult_count} multipliers</div>')

        st.markdown(html, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════
        # ZONE 2 — AUTO-INTELLIGENCE
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sb-zone"><span class="zone-icon">⚡</span>Auto-Intelligence</div>',
                    unsafe_allow_html=True)

        ai_on = st.toggle(
            "Enable Auto-Intelligence",
            value=st.session_state.auto_intelligence,
            help=(
                "When enabled, VEDUTA uses its Auto-Intelligence engine to generate plain-English "
                "executive briefings below each chart and table — interpreting your "
                "actual results, not generic descriptions. Each insight is generated "
                "fresh from your live data."
            ),
        )
        if ai_on != st.session_state.auto_intelligence:
            st.session_state.auto_intelligence = ai_on
            st.session_state.ai_insights_cache = {}  # clear cache on toggle

        if ai_on:
            st.markdown(
                '<div style="font-size:.6rem;color:#C8974A;padding:.25rem 0 .5rem">'
                'AI briefings will appear below each section after running a forecast. '
                'Each insight reads your live numbers.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Clear Insight Cache", use_container_width=True):
                st.session_state.ai_insights_cache = {}
                st.success("Cache cleared — insights will regenerate.")

        # ════════════════════════════════════════════════════════════════════
        # ZONE 3 — BASELINE FORECAST CONTROLS
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sb-zone"><span class="zone-icon">◎</span>Baseline Forecast</div>',
                    unsafe_allow_html=True)

        with st.form("baseline_controls", border=False):

            # Model Tier
            _tiers = ["enterprise","pro","essentials"]
            _cur   = st.session_state.get("sentinel_active_tier","enterprise").lower()
            tier   = st.selectbox(
                "Model Tier",
                _tiers,
                index=_tiers.index(_cur) if _cur in _tiers else 0,
                format_func=lambda x: {
                    "enterprise": "Enterprise  (19 models)",
                    "pro":        "Pro  (13 models)",
                    "essentials": "Essentials  (10 models)",
                }[x],
                help="**Enterprise** — Full model suite including neural/ensemble\n\n"
                     "**Pro** — Balanced accuracy and speed\n\n"
                     "**Essentials** — Core statistical models only",
            )

            # Horizon + Backtest
            c1, c2 = st.columns(2)
            with c1:
                horizon = st.number_input("Forecast Horizon", min_value=1,
                    max_value=60, value=st.session_state.forecast_horizon or 12,
                    help="Periods to forecast forward.")
            with c2:
                backtest = st.number_input("Backtest Window", min_value=4,
                    max_value=36, value=st.session_state.backtest_horizon or 12,
                    help="Held-out periods for MASE evaluation.")

            # CI Level
            ci_options = {0.80:"80%  (Narrow)", 0.90:"90%", 0.95:"95%  (Standard)", 0.99:"99%  (Wide)"}
            ci_level = st.selectbox(
                "Confidence Interval",
                list(ci_options.keys()),
                index=list(ci_options.keys()).index(
                    st.session_state.ci_level if st.session_state.ci_level in ci_options else 0.95),
                format_func=lambda x: ci_options[x],
                help="Width of the forecast confidence band. Wider = more conservative.",
            )

            # Backtest Strategy
            bt_strategy = st.selectbox(
                "Backtest Strategy",
                ["expanding","rolling"],
                index=0 if st.session_state.backtest_strategy=="expanding" else 1,
                format_func=lambda x: {
                    "expanding": "Expanding Window  (default)",
                    "rolling":   "Rolling Window  (fixed size)",
                }[x],
                help="**Expanding** — grows training set each fold, mimics real-world deployment\n\n"
                     "**Rolling** — fixed window size, tests adaptability to recent data",
            )

            # Ensemble Weight Method
            ew_options = {
                "mase":  "MASE-Weighted  ✓ Active",
                "equal": "Equal Weights  (planned)",
            }
            ew_method = st.selectbox(
                "Ensemble Weights",
                list(ew_options.keys()),
                index=0,
                format_func=lambda x: ew_options[x],
                help="**MASE-Weighted** — Each model's contribution is proportional to its "
                     "inverse backtest MASE. Better-performing models dominate the ensemble. "
                     "This is the statistically rigorous default and the only active option.\n\n"
                     "**Equal Weights** — Planned for a future engine release.",
            )

            # Outlier Sensitivity
            os_options = {
                "none":   "None  (raw data)",
                "low":    "Low  (flag only)",
                "medium": "Medium  (cap at 3σ)",
                "high":   "High  (cap at 2σ)",
            }
            outlier_sens = st.selectbox(
                "Outlier Sensitivity",
                list(os_options.keys()),
                index=list(os_options.keys()).index(
                    st.session_state.outlier_sensitivity
                    if st.session_state.outlier_sensitivity in os_options else "medium"),
                format_func=lambda x: os_options[x],
                help="Pre-processing gate applied before the engine runs.\n\n"
                     "**None** — pass raw data through unchanged\n\n"
                     "**Low** — identify outliers but do not remove\n\n"
                     "**Medium** — winsorise at ±3 standard deviations\n\n"
                     "**High** — winsorise at ±2 standard deviations",
            )

            # Analyst Mode
            analyst_mode = st.toggle("Analyst Mode",
                value=st.session_state.analyst_mode,
                help="Enables extended diagnostics and model-level detail in all tab outputs.")

            run_btn = st.form_submit_button("RUN FORECAST",
                use_container_width=True, type="primary")

        if run_btn:
            _run_forecast(tier, horizon, backtest, ci_level,
                          bt_strategy, ew_method, outlier_sens, analyst_mode)

        # ════════════════════════════════════════════════════════════════════
        # ZONE 4 — MACRO VARIABLES
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sb-zone"><span class="zone-icon">◈</span>Macro Variables</div>',
                    unsafe_allow_html=True)

        macro_mode = st.radio(
            "Mode",
            ["Tier A — FRED Regressors", "Tier B — Multipliers"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # ── Tier A — FRED API ─────────────────────────────────────────────
        if macro_mode == "Tier A — FRED Regressors":
            st.markdown(
                '<div style="font-size:.6rem;color:#4b5e80;margin:.15rem 0 .4rem">'
                'Live FRED data fed into SARIMAX + Prophet as exogenous regressors.</div>',
                unsafe_allow_html=True,
            )

            fred_key = st.text_input(
                "FRED API Key",
                value=st.session_state.macro_fred_key or "",
                type="password",
                placeholder="fred.stlouisfed.org — free key",
                help="Free key from fred.stlouisfed.org/docs/api/api_key.html",
            )
            if fred_key:
                st.session_state.macro_fred_key = fred_key

            # Build flat options list: "Category — Label (UNIT) [SID]"
            all_options = {}
            for sid, meta in FRED_CATALOGUE.items():
                display = f"{meta['category']} — {meta['label']} ({meta['unit']})"
                all_options[display] = sid

            # Current selection as display strings
            current_sids = st.session_state.macro_selected_series or []
            sid_to_display = {v: k for k, v in all_options.items()}
            current_display = [sid_to_display[s] for s in current_sids if s in sid_to_display]

            selected_display = st.multiselect(
                "Select FRED Series",
                options=list(all_options.keys()),
                default=current_display,
                help="Select macro series to use as model regressors.",
                placeholder="Choose series…",
            )
            selected_sids = [all_options[d] for d in selected_display]
            st.session_state.macro_selected_series = selected_sids

            if selected_sids:
                st.markdown(
                    f'<div class="macro-badge">{len(selected_sids)} series selected</div>',
                    unsafe_allow_html=True,
                )

            col_f, col_c = st.columns(2)
            with col_f:
                if st.button("Fetch FRED Data", use_container_width=True,
                             disabled=not (fred_key and selected_sids)):
                    if st.session_state.committed_df is None:
                        st.error("Commit data first.")
                    else:
                        with st.spinner("Fetching from FRED…"):
                            exog, errs = build_exog_df(
                                st.session_state.committed_df,
                                selected_sids, fred_key,
                            )
                        if exog is not None:
                            st.session_state.macro_exog_df      = exog
                            st.session_state.macro_vars_enabled = True
                            st.success(f"Fetched {exog.shape[1]} series, {len(exog)} periods")
                        for e in errs:
                            st.warning(e)
            with col_c:
                if st.button("Clear FRED", use_container_width=True):
                    st.session_state.macro_exog_df         = None
                    st.session_state.macro_vars_enabled    = False
                    st.session_state.macro_selected_series = []
                    st.info("Cleared.")

            if st.session_state.macro_exog_df is not None:
                st.markdown(
                    '<div class="macro-badge">◈ FRED data loaded · '
                    'Ready for engine integration — contact your platform engineer to enable '
                    'exogenous regressor support in sentinel_engine.run_all_models()</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div style="font-size:.58rem;color:#4b5e80;margin-top:.4rem;'
                'padding:.4rem .5rem;border:1px solid #1e2a45;border-radius:4px;">'
                '⚠ Tier A status: FRED fetch is live. Regressor injection into model '
                'training requires a Foresight engine update to accept exog_df. '
                'Tier B multipliers are fully active today.</div>',
                unsafe_allow_html=True,
            )

        # ── Tier B — Multipliers ──────────────────────────────────────────
        else:
            st.markdown(
                '<div style="font-size:.6rem;color:#4b5e80;margin:.15rem 0 .4rem">'
                'Set macro assumptions — VEDUTA applies calibrated shocks to the baseline.</div>',
                unsafe_allow_html=True,
            )

            tier_b_enabled = st.toggle(
                "Enable Macro Multipliers",
                value=st.session_state.macro_multiplier_enabled,
            )
            st.session_state.macro_multiplier_enabled = tier_b_enabled

            if tier_b_enabled:
                multipliers = {}
                for var_name, preset in TIER_B_PRESETS.items():
                    shock = st.number_input(
                        var_name,
                        value=float(st.session_state.macro_multipliers.get(var_name, 0.0)),
                        step=0.1,
                        help=preset["description"],
                        key=f"tb_{var_name}",
                    )
                    multipliers[var_name] = shock

                st.session_state.macro_multipliers = multipliers

                total_impact = sum(
                    shock * TIER_B_PRESETS[v]["multiplier"] / 100.0
                    for v, shock in multipliers.items() if v in TIER_B_PRESETS
                )
                impact_color = "#22c55e" if total_impact > 0 else "#ef4444" if total_impact < 0 else "#64748b"
                st.markdown(
                    f'<div style="background:#1B2A40;border:1px solid #243347;'
                    f'border-radius:8px;padding:.6rem;margin:.4rem 0;text-align:center">'
                    f'<div style="font-size:.56rem;color:#4b5e80;letter-spacing:.1em;'
                    f'text-transform:uppercase;margin-bottom:.2rem">Net Macro Impact</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{impact_color}">'
                    f'{total_impact*100:+.2f}%</div></div>',
                    unsafe_allow_html=True,
                )

                col_a, col_r = st.columns(2)
                with col_a:
                    if st.button("Apply to Forecast", use_container_width=True,
                                 type="primary",
                                 disabled=st.session_state.sentinel_primary_df is None):
                        baseline = st.session_state.sentinel_primary_df
                        adjusted = apply_tier_b_multipliers(baseline, multipliers)
                        st.session_state.scenario_forecast_df = adjusted
                        st.session_state.scenario_state = {
                            "enabled": True,
                            "params": {
                                "type":             "Macro Multipliers",
                                "shock_pct":        total_impact * 100,
                                "trend_adjust_pct": 0,
                                "recovery_periods": 0,
                            },
                            "applied_signature": st.session_state.run_signature,
                        }
                        st.session_state.audit_log.append({
                            "event":           "macro_multiplier_apply",
                            "timestamp":       datetime.utcnow().isoformat(),
                            "multipliers":     {k: v for k, v in multipliers.items() if v != 0},
                            "net_impact_pct":  total_impact * 100,
                        })
                        st.success(f"Applied — net {total_impact*100:+.2f}%")
                with col_r:
                    if st.button("Reset", use_container_width=True):
                        st.session_state.macro_multipliers = {}
                        st.rerun()

        # ════════════════════════════════════════════════════════════════════
        # ZONE 5 — SCENARIO SIMULATION
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sb-zone"><span class="zone-icon">▲</span>Scenario Simulation</div>',
                    unsafe_allow_html=True)

        forecast_ready = st.session_state.sentinel_primary_df is not None

        if not forecast_ready:
            st.markdown(
                '<div style="font-size:.62rem;color:#4b5e80;padding:.25rem 0">'
                'Run a forecast to unlock scenario controls.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.form("scenario_controls", border=False):
                scenario_type = st.selectbox(
                    "Scenario Type",
                    ["Shock","Trend Shift","Ramp Recovery","Shock + Recovery"],
                    help="**Shock** — Immediate % impact\n\n"
                         "**Trend Shift** — Progressive growth/decline\n\n"
                         "**Ramp Recovery** — Shock with gradual return\n\n"
                         "**Shock + Recovery** — Sharp impact with recovery arc",
                )
                c1, c2 = st.columns(2)
                with c1:
                    shock = st.number_input("Shock (%)", value=0.0, step=1.0)
                with c2:
                    trend_adj = st.number_input("Trend Adj (%)", value=0.0, step=0.5)
                recovery = st.number_input("Recovery Periods", min_value=0,
                                           max_value=60, value=0)
                c_apply, c_clear = st.columns(2)
                with c_apply:
                    apply_btn = st.form_submit_button("APPLY", use_container_width=True,
                                                      type="primary")
                with c_clear:
                    clear_btn = st.form_submit_button("CLEAR", use_container_width=True)

            if apply_btn:
                scenario_series = apply_scenario(
                    st.session_state.sentinel_primary_df,
                    scenario_type    = scenario_type,
                    shock_pct        = shock,
                    trend_adjust_pct = trend_adj,
                    recovery_periods = recovery,
                )
                st.session_state.scenario_state = {
                    "enabled": True,
                    "params": {"type": scenario_type, "shock_pct": shock,
                               "trend_adjust_pct": trend_adj, "recovery_periods": recovery},
                    "applied_signature": st.session_state.run_signature,
                }
                st.session_state.scenario_forecast_df = scenario_series
                st.session_state.audit_log.append({
                    "event": "scenario_apply", "timestamp": datetime.utcnow().isoformat(),
                    "params": st.session_state.scenario_state["params"],
                })
                st.success(f"Scenario applied — {scenario_type}")

            if clear_btn:
                st.session_state.scenario_state = {
                    "enabled": False, "params": None, "applied_signature": None}
                st.session_state.scenario_forecast_df = None
                st.info("Scenario cleared.")

        # ════════════════════════════════════════════════════════════════════
        # ZONE 6 — QUICK REFERENCE
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="sb-zone"><span class="zone-icon">≡</span>Quick Reference</div>',
                    unsafe_allow_html=True)

        sig = st.session_state.run_signature
        if sig:
            ref_html = "".join([
                _row(k, str(v)[:20] if len(str(v)) > 20 else str(v))
                for k, v in {
                    "Run ID":     sig.get("run_id","—")[-12:],
                    "Tier":       sig.get("active_tier","—").title(),
                    "Horizon":    f"{sig.get('forecast_horizon','—')}p",
                    "Backtest":   f"{sig.get('backtest_horizon','—')}p",
                    "CI":         f"{int(sig.get('ci_level',0.95)*100)}%",
                    "Weights":    sig.get("ensemble_weights","—"),
                    "Outliers":   sig.get("outlier_sensitivity","—").title(),
                    "Winner":     sig.get("winner","—"),
                }.items()
            ])
            st.markdown(ref_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-size:.62rem;color:#4b5e80">No run signature yet.</div>',
                unsafe_allow_html=True,
            )

        # Footer
        st.markdown(
            '<div style="margin-top:2rem;padding-top:.75rem;border-top:1px solid #243347;'
            'font-size:.55rem;color:#243347;text-align:center;letter-spacing:.06em">'
            'VEDUTA · Foresight Engine v1.0<br>'
            'Manufacturing & Supply Chain Intelligence'
            '</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FORECAST RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def _run_forecast(tier, horizon, backtest, ci_level,
                  bt_strategy, ew_method, outlier_sens, analyst_mode):
    """Execute forecast and populate all session state keys."""

    if st.session_state.committed_df is None:
        st.error("No committed data. Upload and commit data on the Home tab.")
        return
    if st.session_state.data_frequency is None:
        st.error("Frequency not detected. Re-commit data on the Home tab.")
        return

    # Pre-process: outlier winsorisation
    df = st.session_state.committed_df.copy()
    df, outlier_log = _apply_outlier_filter(df, outlier_sens)

    st.session_state.forecast_status      = "Running"
    st.session_state.sentinel_active_tier = tier
    st.session_state.ci_level             = ci_level
    st.session_state.backtest_strategy    = bt_strategy
    st.session_state.ensemble_weight_method = ew_method
    st.session_state.outlier_sensitivity  = outlier_sens
    st.session_state.analyst_mode         = analyst_mode

    with st.spinner("Foresight Engine running…"):
        try:
            exog = st.session_state.macro_exog_df if st.session_state.macro_vars_enabled else None
            bundle = run_forecast(
                df                     = df,
                frequency              = st.session_state.data_frequency,
                backtest_horizon       = backtest,
                forecast_horizon       = horizon,
                active_tier            = tier,
                confidence_level       = ci_level,
                exog_df                = exog,
                backtest_strategy      = bt_strategy,
                ensemble_weight_method = ew_method,
            )
        except Exception as e:
            st.session_state.forecast_status = "Failed"
            st.error(f"Engine error: {e}")
            return

    leg = bundle["legacy"]

    # ── Legacy session keys ───────────────────────────────────────────────
    st.session_state.latest_metrics         = leg["metrics_df"]
    st.session_state.latest_forecasts        = leg["forecasts"]
    st.session_state.latest_intervals        = leg["intervals"]
    st.session_state.latest_model_name       = leg["winner"]
    st.session_state.latest_forecast_df      = leg["forecast_viz_df"]
    st.session_state.latest_forecast_series  = leg["winner_forecast"]

    # ── Sentinel session keys ─────────────────────────────────────────────
    st.session_state.sentinel_results        = bundle["sentinel"]
    st.session_state.sentinel_engine_version = bundle["engine_meta"].get("engine_version","2.0.0")
    st.session_state.sentinel_primary_df     = leg["primary_df"]
    st.session_state.sentinel_stacked_df     = leg["stacked_df"]
    st.session_state.sentinel_cert_metadata  = bundle["cert_metadata"]
    st.session_state.sentinel_run_metadata   = bundle["engine_meta"]

    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    st.session_state.run_signature = {
        "run_id":               run_id,
        "timestamp":            datetime.utcnow().isoformat(),
        "data_fingerprint":     st.session_state.committed_meta["data_fingerprint"],
        "forecast_horizon":     horizon,
        "backtest_horizon":     backtest,
        "active_tier":          tier,
        "ci_level":             ci_level,
        "backtest_strategy":    bt_strategy,
        "ensemble_weights":     ew_method,
        "outlier_sensitivity":  outlier_sens,
        "analyst_mode":         analyst_mode,
        "engine_version":       st.session_state.sentinel_engine_version,
        "winner":               leg["winner"],
        "macro_fred_series":    st.session_state.macro_selected_series,
        "macro_multipliers_active": st.session_state.macro_multiplier_enabled,
        "outlier_log":          outlier_log,
    }

    st.session_state.forecast_horizon      = horizon
    st.session_state.backtest_horizon       = backtest
    st.session_state.last_run_timestamp     = datetime.utcnow().isoformat()
    st.session_state.forecast_status        = "Complete"
    st.session_state.ai_insights_cache      = {}  # invalidate AI cache on new run

    st.session_state.audit_log.append({
        "event": "forecast_run", "timestamp": datetime.utcnow().isoformat(),
        "run_id": run_id, "tier": tier, "horizon": horizon,
        "backtest": backtest, "ci_level": ci_level,
        "engine_version": st.session_state.sentinel_engine_version,
        "winner": leg["winner"],
    })

    models_ok = leg["metrics_df"]["status"].eq("OK").sum() if leg["metrics_df"] is not None else "?"
    st.success(
        f"✓ Complete — {tier.title()} · {horizon}p · "
        f"{models_ok} models OK · CI {int(ci_level*100)}%"
    )


def _apply_outlier_filter(df: pd.DataFrame, sensitivity: str):
    """
    Winsorise the value column based on outlier sensitivity setting.
    Returns (filtered_df, log_dict).
    """
    log = {"sensitivity": sensitivity, "capped": 0}

    if sensitivity == "none":
        return df, log

    vals = df["value"].copy()
    mu, sigma = vals.mean(), vals.std()

    cap = {"low": None, "medium": 3.0, "high": 2.0}.get(sensitivity)

    if cap is None:  # low — flag only, don't modify
        outliers = ((vals - mu).abs() > 3 * sigma).sum()
        log["flagged"] = int(outliers)
        return df, log

    lower = mu - cap * sigma
    upper = mu + cap * sigma
    capped = ((vals < lower) | (vals > upper)).sum()
    df = df.copy()
    df["value"] = vals.clip(lower=lower, upper=upper)
    log["capped"] = int(capped)
    log["bounds"] = [round(lower, 2), round(upper, 2)]

    return df, log
