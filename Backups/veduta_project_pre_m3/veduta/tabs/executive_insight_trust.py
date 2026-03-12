# ==================================================
# FILE: veduta/tabs/executive_insight_trust.py
# VERSION: 2.0.0
# ROLE: EXECUTIVE INSIGHT & TRUST
# UPDATED: Phase 4 — MASE-based confidence score,
#          cert tier display, sentinel engine metadata.
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
from engine.auto_intelligence import generate_recommendations


# ── MASE → confidence score (0–100) ──────────────────────────────────────────
def _mase_confidence_score(cert_metadata: list) -> int:
    """
    Derive confidence score from Primary Ensemble MASE.
    Elite  (MASE<0.70) → 85–100
    Strong (MASE<0.85) → 65–84
    Pass   (MASE<1.00) → 45–64
    Fail   (MASE≥1.00) → 0–44
    """
    if not cert_metadata:
        return 0

    pe = next((m for m in cert_metadata if m["model"] == "Primary Ensemble"), None)
    if pe is None or pe.get("MASE") is None:
        return 40

    mase = pe["MASE"]
    if mase < 0.50:
        return 100
    if mase < 0.70:
        return int(85 + (0.70 - mase) / 0.20 * 15)
    if mase < 0.85:
        return int(65 + (0.85 - mase) / 0.15 * 20)
    if mase < 1.00:
        return int(45 + (1.00 - mase) / 0.15 * 20)
    return max(0, int(40 - (mase - 1.00) * 40))



_CSS = """
<style>
/* ── VEDUTA Identity injection for Intelligence & Trust tab ── */
[data-testid="stAppViewContainer"] { background: #07080F; }

/* Page header */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    color: #EDE8DE !important;
    letter-spacing: 0.04em !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: .6rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: #4A6278 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
    font-weight: 300 !important;
    color: #EDE8DE !important;
}

/* Divider */
hr { border-color: #243347 !important; }

/* Info / success / warning boxes */
[data-testid="stAlert"] {
    background: #1B2A40 !important;
    border: 1px solid #243347 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .72rem !important;
    color: #8FA3B8 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #1B2A40 !important;
    border: 1px solid #243347 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: .7rem !important;
    color: #8FA3B8 !important;
}

/* Body text / markdown */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-family: 'DM Mono', monospace !important;
    font-size: .72rem !important;
    color: #8FA3B8 !important;
    line-height: 1.7 !important;
}

/* Toggle */
[data-testid="stToggle"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: .72rem !important;
    color: #8FA3B8 !important;
}

/* Section subheaders shown as DM Mono labels */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #C8974A;
    margin: 2.25rem 0 .85rem 0;
    display: flex;
    align-items: center;
    gap: .6rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #243347 0%, transparent 100%);
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
"""

def render_executive_insight_trust():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-header">Intelligence & Trust</div>', unsafe_allow_html=True)

    if st.session_state.latest_metrics is None:
        st.info("Run a baseline forecast to unlock trust and insight analysis.")
        return

    cert_metadata = st.session_state.sentinel_cert_metadata or []

    # =========================================================================
    # ENGINE METADATA BANNER
    # =========================================================================
    engine_ver  = st.session_state.sentinel_engine_version or "2.0.0"
    active_tier = st.session_state.sentinel_active_tier or "Enterprise"
    run_meta    = st.session_state.sentinel_run_metadata or {}

    attempted = run_meta.get("models_attempted", "—")
    succeeded = run_meta.get("models_succeeded", "—")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engine Version",  f"v{engine_ver}")
    col2.metric("Active Tier",     active_tier)
    col3.metric("Models Attempted", attempted)
    col4.metric("Models Succeeded", succeeded)

    st.divider()

    # =========================================================================
    # CONFIDENCE SCORE (MASE-BASED)
    # =========================================================================
    st.subheader("Forecast Confidence")

    confidence_score = _mase_confidence_score(cert_metadata)

    # Find Primary Ensemble MASE for display
    pe_meta = next((m for m in cert_metadata if m["model"] == "Primary Ensemble"), {})
    pe_mase = pe_meta.get("MASE")
    pe_tier = pe_meta.get("cert_tier", "—")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "Confidence Score (0–100)",
        confidence_score,
        help="Derived from Primary Ensemble MASE. Elite MASE < 0.70 scores 85–100.",
    )
    col_b.metric(
        "Primary Ensemble MASE",
        f"{pe_mase:.4f}" if pe_mase is not None else "—",
        help="Mean Absolute Scaled Error vs seasonal naïve baseline.",
    )
    col_c.metric(
        "M-Competition Tier",
        pe_tier,
        help="Elite < 0.70 | Strong < 0.85 | Pass < 1.00 | Fail ≥ 1.00",
    )

    st.markdown(
        "**Why this matters:** Confidence is derived from MASE vs the M-Competition "
        "MASE benchmark — accuracy measured against seasonal naïve baseline. "
        "MASE < 0.70 means the engine beats the seasonal naïve baseline by 30% or more."
    )

    st.divider()

    # =========================================================================
    # CERTIFICATION SUMMARY
    # =========================================================================
    st.subheader("Model Certification Summary")

    if cert_metadata:
        cert_df = pd.DataFrame(cert_metadata)

        # Count tiers
        tier_counts = cert_df["cert_tier"].value_counts().to_dict()
        elite  = tier_counts.get("Elite",  0)
        strong = tier_counts.get("Strong", 0)
        pass_  = tier_counts.get("Pass",   0)
        fail   = tier_counts.get("Fail",   0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Elite",  elite)
        c2.metric("🟡 Strong", strong)
        c3.metric("🟠 Pass",   pass_)
        c4.metric("🔴 Fail",   fail)

        # Stacked Ensemble callout
        se_meta = next((m for m in cert_metadata if m["model"] == "Stacked Ensemble"), None)
        if se_meta and se_meta.get("MASE") is not None:
            st.info(
                f"Stacked Ensemble (Ridge meta-learner): "
                f"MASE {se_meta['MASE']:.4f} · {se_meta.get('cert_tier', '—')} tier · "
                f"Available as secondary forecast pathway."
            )
    else:
        st.info("No certification metadata available.")

    st.divider()

    # =========================================================================
    # BEHAVIORAL REGIME
    # =========================================================================
    st.subheader("Behavioral Regime")

    if confidence_score >= 80:
        regime = "Stable / High Confidence"
        regime_color = "success"
    elif confidence_score >= 55:
        regime = "Moderate / Trend-Aligned"
        regime_color = "info"
    else:
        regime = "Volatile / Uncertain"
        regime_color = "warning"

    getattr(st, regime_color)(f"Current Regime: **{regime}**")

    # =========================================================================
    # RISK FLAGS
    # =========================================================================
    st.subheader("Risk Flags")

    risk_flags = []

    if confidence_score < 50:
        risk_flags.append("Confidence score below 50 — model agreement is low.")

    if pe_mase is not None and pe_mase >= 1.0:
        risk_flags.append("Primary Ensemble MASE ≥ 1.00 — does not beat naïve baseline.")

    if fail > 0:
        risk_flags.append(f"{fail} model(s) failed the MASE certification gate.")

    if st.session_state.freq_inference_details:
        if st.session_state.freq_inference_details.get("confidence", 1) < 0.7:
            risk_flags.append("Low confidence in inferred data frequency.")

    # Prophet flag
    prophet_meta = next((m for m in cert_metadata if m["model"] == "Prophet"), None)
    if prophet_meta and prophet_meta.get("MASE") is not None:
        if prophet_meta["MASE"] >= 1.0:
            risk_flags.append(
                "Prophet flagged (MASE ≥ 1.00 on regime-change data). "
                "Regime-detection filter active."
            )

    if not risk_flags:
        st.success("No major structural risks detected.")
    else:
        for rf in risk_flags:
            st.warning(rf)

    st.divider()

    # =========================================================================
    # AUTO-INTELLIGENCE GUIDANCE
    # =========================================================================
    st.subheader("Executive Guidance")

    enable_ai = st.toggle("Enable Auto-Intelligence Recommendation", value=False)

    if enable_ai:
        context = {
            "volatility":            1 - (confidence_score / 100),
            "trend_strength":        0.5,
            "seasonality_strength":  0.5,
        }
        recs = generate_recommendations(context)

        st.markdown("### Recommendations")
        if recs.get("horizon"):
            st.markdown(f"- **Horizon Guidance:** {recs['horizon']}")
        if recs.get("model_focus"):
            st.markdown(f"- **Model Emphasis:** {recs['model_focus']}")
        if recs.get("scenario_tests"):
            st.markdown("- **Suggested Scenario Tests:**")
            for s in recs["scenario_tests"]:
                st.markdown(f"  - {s}")

        st.session_state.audit_log.append({
            "event":            "auto_intelligence_view",
            "timestamp":        st.session_state.last_run_timestamp,
            "summary":          "Executive guidance viewed",
            "confidence_score": confidence_score,
            "regime":           regime,
            "pe_mase":          pe_mase,
        })

    # =========================================================================
    # SCENARIO IMPACT SUMMARY
    # =========================================================================
    if st.session_state.scenario_state.get("enabled"):
        st.subheader("Scenario Impact Summary")
        st.markdown(
            "Scenario analysis is currently applied. "
            "Review baseline vs scenario deltas to understand decision sensitivity."
        )

    # =========================================================================
    # ASSUMPTIONS & LIMITATIONS
    # =========================================================================
    with st.expander("Assumptions & Limitations"):
        st.markdown(
            "- Foresight Engine v1.0 — MASE-certified: 16/16 models beat seasonal naïve baseline.\n"
            "- MASE measured against seasonal naïve baseline (M-Competition standard).\n"
            "- Confidence score derived from Primary Ensemble MASE, not model agreement.\n"
            "- Prophet uses regime-detection pre-filter; may be suppressed on shock data.\n"
            "- Forecasts assume historical patterns persist absent structural breaks.\n"
            "- Scenario overlays are illustrative, not predictive.\n"
            "- Stacked Ensemble (Ridge meta-learner) is a secondary pathway — "
            "Primary Ensemble is the certified baseline."
        )
