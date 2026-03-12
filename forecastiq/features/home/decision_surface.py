# ==================================================
# FILE: features/home/decision_surface.py
# ROLE: EXECUTIVE VERDICT & FINANCIAL IMPACT (LOCKED)
# STATUS: GOVERNANCE-GRADE / SCENARIO-AWARE / INTENT-AWARE
#
# GOVERNANCE:
# - READS st.session_state only
# - NO forecasting logic
# - NO mutation (EXCEPT: publishing verdict payload for intelligence layer)
# - SINGLE authoritative forecast context per decision
# ==================================================

import streamlit as st
import numpy as np


# ==================================================
# EXECUTIVE VERDICT (INTENT-AWARE 5-TIER RISK LADDER)
# ==================================================

def render_executive_verdict():
    """
    Executive decision verdict based on ACTIVE forecast context
    and declared Executive Intent.

    Decision hierarchy:
    1. Scenario downside risk (mathematical)
    2. Executive macro intent (verdict caps)
    3. Model confidence (baseline)
    4. Trend direction (active context)
    """

    scenario_df = st.session_state.get("scenario_forecast_df")
    baseline_df = st.session_state.get("latest_forecast_df")
    metrics = st.session_state.get("latest_metrics")
    winner = st.session_state.get("latest_model_name")
    audit = st.session_state.get("active_scenario_audit", {})
    intent = st.session_state.get("executive_intent", {})

    if baseline_df is None or metrics is None or winner is None:
        return

    # --------------------------------------------------
    # SELECT AUTHORITATIVE FORECAST CONTEXT
    # --------------------------------------------------
    if scenario_df is not None:
        df = scenario_df
        context_label = audit.get("scenario_name", "Scenario")
    else:
        df = baseline_df
        context_label = "Baseline"

    # --------------------------------------------------
    # MODEL CONFIDENCE (BASELINE — STABLE)
    # --------------------------------------------------
    rmse_vals = metrics.loc[metrics["model"] == winner, "RMSE"].values
    rmse = float(rmse_vals[0]) if len(rmse_vals) else np.nan

    mean_actual = baseline_df["actual"].dropna().mean()
    confidence_score = (
        max(0, 100 - (rmse / mean_actual) * 100)
        if mean_actual and not np.isnan(rmse)
        else 0
    )

    # --------------------------------------------------
    # TREND SIGNAL (ACTIVE CONTEXT)
    # --------------------------------------------------
    future_fc = df[df["is_future"] == True]["forecast"]
    trend = future_fc.iloc[-1] - future_fc.iloc[0] if len(future_fc) >= 2 else 0

    # --------------------------------------------------
    # SCENARIO DOWNSIDE (% VS BASELINE)
    # --------------------------------------------------
    downside_pct = 0.0
    if scenario_df is not None:
        try:
            b = baseline_df[baseline_df["is_future"] == True]["forecast"].sum()
            s = scenario_df[scenario_df["is_future"] == True]["forecast"].sum()
            if b != 0:
                downside_pct = (s - b) / b * 100
        except Exception:
            downside_pct = 0.0

    # --------------------------------------------------
    # BASE VERDICT FROM MATH (5-TIER)
    # --------------------------------------------------
    if downside_pct <= -12:
        verdict = "HOLD / REASSESS"
        color = "🔴"
        posture = "Severe Risk"
        instruction = "Stop new commitments. Replan assumptions. Executive review required."

    elif downside_pct <= -10:
        verdict = "HOLD — MITIGATE"
        color = "🟠"
        posture = "High Risk"
        instruction = "Pause expansion. Activate mitigation plan. Tighten cost and exposure."

    elif downside_pct <= -7:
        verdict = "PROCEED WITH CAUTION"
        color = "🟡"
        posture = "Moderate Risk"
        instruction = "Proceed selectively. Scenario planning and contingencies advised."

    elif downside_pct <= -4:
        verdict = "GO — MONITOR"
        color = "🟢🟡"
        posture = "Early Warning"
        instruction = "Proceed, but monitor leading indicators closely."

    else:
        verdict = "GO"
        color = "🟢"
        posture = "Normal Risk"
        instruction = "Proceed as planned. Standard monitoring applies."

    # --------------------------------------------------
    # EXECUTIVE INTENT VERDICT CAPS (AUTHORITATIVE)
    # --------------------------------------------------
    macro = intent.get("macro_regime", "baseline")
    severity = intent.get("severity", "none")
    risk_posture = intent.get("risk_posture", "balanced")

    intent_note = None

    if macro == "recession":
        if severity == "light" and verdict == "GO":
            verdict, color = "GO — MONITOR", "🟢🟡"
            posture = "Executive-Capped Risk"
            intent_note = "Light recession selected — optimism capped by executive intent."

        elif severity == "moderate" and verdict in ("GO", "GO — MONITOR"):
            verdict, color = "PROCEED WITH CAUTION", "🟡"
            posture = "Executive-Capped Risk"
            intent_note = "Moderate recession selected — caution enforced."

        elif severity == "severe" and verdict not in ("HOLD — MITIGATE", "HOLD / REASSESS"):
            verdict, color = "HOLD — MITIGATE", "🟠"
            posture = "Executive-Capped Risk"
            intent_note = "Severe recession selected — risk floor applied."

    if risk_posture == "defensive" and verdict == "GO":
        verdict, color = "GO — MONITOR", "🟢🟡"
        posture = "Defensive Bias"
        intent_note = "Defensive posture selected — downgraded for capital preservation."

    # --------------------------------------------------
    # 🔐 PUBLISH VERDICT PAYLOAD (FOR AUTO-INTELLIGENCE)
    # --------------------------------------------------
    st.session_state["executive_verdict_payload"] = {
        "verdict": verdict,
        "risk_posture": posture,
        "downside_pct": round(downside_pct, 2),
        "confidence_score": round(confidence_score, 1),
        "trend": "upward" if trend > 0 else "flat_or_down",
        "context": context_label,
        "intent_applied": bool(intent_note),
        "intent_note": intent_note,
    }

    # --------------------------------------------------
    # RENDER
    # --------------------------------------------------
    st.markdown("## Executive Decision Verdict")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", f"{color} {verdict}")
    c2.metric("Risk Posture", posture)
    c3.metric("Scenario Downside", f"{downside_pct:.1f}%")
    c4.metric("Model Confidence", f"{confidence_score:.1f}%")

    st.caption(f"Decision context: **{context_label}**")
    st.caption(f"Trend signal: {'Upward' if trend > 0 else 'Flat / Down'}")
    st.caption(f"Executive guidance: {instruction}")

    # ✅ ALWAYS SHOW INTENT BANNER WHEN RECESSION IS ACTIVE
    if macro == "recession":
        if intent_note:
            st.info(f"Executive intent applied: {intent_note}")
        else:
            st.info(
                "Executive intent applied: Recession scenario active — "
                "verdict reflects executive risk framing even where model signals remain resilient."
            )


# ==================================================
# FINANCIAL IMPACT (SCENARIO-AWARE — UNCHANGED)
# ==================================================

def render_financial_impact():
    """
    Financial impact based on ACTIVE forecast context.
    """

    scenario_df = st.session_state.get("scenario_forecast_df")
    baseline_df = st.session_state.get("latest_forecast_df")
    audit = st.session_state.get("active_scenario_audit", {})

    df = scenario_df if scenario_df is not None else baseline_df
    if df is None:
        return

    context_label = audit.get("scenario_name", "Baseline")

    st.markdown("## Financial Impact Scenarios")

    future_df = df[df["is_future"] == True]
    if future_df.empty:
        st.caption("Insufficient forecast horizon for financial impact analysis.")
        return

    avg_value = future_df["forecast"].mean()
    horizon = len(future_df)

    base = avg_value * horizon
    upside = base * 1.10
    downside = base * 0.90

    c1, c2, c3 = st.columns(3)
    c1.metric("Base Case ($)", f"${base:,.0f}")
    c2.metric("Upside (+10%)", f"${upside:,.0f}")
    c3.metric("Downside (-10%)", f"${downside:,.0f}")

    st.caption(f"Financials reflect **{context_label}** forecast context.")
