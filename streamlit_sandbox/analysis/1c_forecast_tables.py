# ==================================================
# FILE: analysis/forecast_tables.py
# ROLE: EXECUTIVE FORECAST RESULT RENDERER (FORTUNE 100 HARDENED + CERTIFICATION AWARE)
# ==================================================
# GOVERNANCE:
# - Safe against non-model entries (e.g., _failures list)
# - Schema validation enforced
# - No ranking or auto-selection
# - Failure panel rendered explicitly
# - Defensive rendering at every boundary
# - Certification eligibility enforced (36 observations minimum)
# ==================================================

import streamlit as st
import pandas as pd
import altair as alt

from analysis.executive_narratives import generate_executive_narrative


MODEL_RANKING_DISCLOSURE = (
    "Model rankings are provisional and reflect decision-readiness only."
)

CERTIFICATION_MIN_OBS = 36

REQUIRED_COLUMNS = {"date", "forecast"}
OPTIONAL_COLUMNS = {"actual", "ci_low", "ci_high", "error_pct"}


# ==================================================
# SAFE CHART BUILDER
# ==================================================

def _forecast_chart(df: pd.DataFrame) -> alt.Chart:

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Forecast output is not a dataframe.")

    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError("Forecast dataframe missing required columns.")

    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date"),
    )

    layers = []

    # Confidence interval band (if valid)
    if {"ci_low", "ci_high"}.issubset(df.columns):
        if df[["ci_low", "ci_high"]].notna().any().any():
            layers.append(
                base.mark_area(opacity=0.2, color="#d62728")
                .encode(y="ci_low:Q", y2="ci_high:Q")
            )

    # Actuals
    if "actual" in df.columns and df["actual"].notna().any():
        layers.append(
            base.mark_line(color="#1f77b4", strokeWidth=2)
            .encode(y="actual:Q")
        )

    # Forecast (required)
    layers.append(
        base.mark_line(color="#d62728", strokeWidth=2)
        .encode(y="forecast:Q")
    )

    return alt.layer(*layers).properties(height=280)


# ==================================================
# READINESS BADGE RENDERER
# ==================================================

def _render_readiness_badge(metadata: dict, df: pd.DataFrame):

    readiness = metadata.get("readiness_tier")

    # Count historical observations
    hist_obs = df["actual"].notna().sum() if "actual" in df.columns else 0

    st.markdown("### Certification Readiness")

    if readiness == "Primary":
        st.success("🟢 Certified Primary Model — Eligible for executive decision use.")
        return

    # Enforce 36 observation rule
    if hist_obs < CERTIFICATION_MIN_OBS:
        st.error(
            f"🔴 Insufficient Data — {hist_obs} observations detected. "
            f"Minimum {CERTIFICATION_MIN_OBS} required for certification eligibility."
        )
        st.caption(
            "Certification standards require a minimum of 36 monthly observations "
            "to ensure regime robustness and statistical reliability."
        )
        return

    # Eligible but not certified
    st.warning(
        f"🟡 Eligible for Certification — {hist_obs} observations detected."
    )
    st.caption(
        "Model meets minimum 36-observation threshold but is not currently "
        "designated as a certified Primary model."
    )


# ==================================================
# FAILURE PANEL
# ==================================================

def _render_failure_panel(failures: list):

    if not failures:
        return

    st.markdown("## Model Failure Log")
    st.error(
        "One or more models failed during execution. "
        "Failures are isolated and logged. Successful models remain valid."
    )

    for record in failures:
        model = record.get("model", "Unknown")
        error = record.get("error_message", "No message")
        exc_type = record.get("exception_type", "Unknown")

        with st.expander(f"{model} — {exc_type}", expanded=False):
            st.code(error)


# ==================================================
# MAIN RENDERER
# ==================================================

def render_forecast_tables(results: dict):

    if not isinstance(results, dict):
        st.error("Forecast results structure invalid.")
        return

    st.subheader("Forecast Results by Model")
    st.caption(MODEL_RANKING_DISCLOSURE)

    failures = results.get("_failures", [])
    if isinstance(failures, list):
        _render_failure_panel(failures)

    for model_name, payload in results.items():

        if not isinstance(payload, dict):
            continue

        with st.expander(model_name, expanded=False):

            if payload.get("status") != "success":
                st.error(f"Model failed: {payload.get('error')}")
                continue

            df: pd.DataFrame = payload.get("forecast_df")

            if not isinstance(df, pd.DataFrame) or df.empty:
                st.warning("No forecast output produced.")
                continue

            if not REQUIRED_COLUMNS.issubset(df.columns):
                st.error("Forecast output schema invalid.")
                continue

            metrics = payload.get("metrics") or {}
            diagnostics = payload.get("diagnostics") or {}
            executive_assessment = payload.get("executive_assessment") or {}
            metadata = payload.get("metadata") or {}

            # --------------------------------------------------
            # READINESS BADGE
            # --------------------------------------------------

            _render_readiness_badge(metadata, df)

            # --------------------------------------------------
            # EXECUTIVE HEADER
            # --------------------------------------------------

            st.markdown(f"### Executive Assessment — {model_name}")
            st.markdown(
                f"**Confidence Posture:** "
                f"{executive_assessment.get('confidence_posture', 'Not Assessed')}"
            )

            # --------------------------------------------------
            # EXECUTIVE NARRATIVE
            # --------------------------------------------------

            try:
                narrative = generate_executive_narrative(
                    model_name=model_name,
                    metrics=metrics,
                    diagnostics=diagnostics,
                    executive_assessment=executive_assessment,
                )
            except Exception:
                narrative = {}

            st.markdown(f"**Summary:** {narrative.get('summary', 'N/A')}")

            if narrative.get("insights"):
                st.markdown("**Key Insights**")
                for bullet in narrative["insights"]:
                    st.markdown(f"- {bullet}")

            if narrative.get("decision_guidance"):
                st.info(narrative["decision_guidance"])

            if narrative.get("risk_flags"):
                st.warning("Risk Flags: " + ", ".join(narrative["risk_flags"]))

            # --------------------------------------------------
            # CI DISCLOSURE
            # --------------------------------------------------

            if metadata.get("ci_note"):
                st.info(metadata["ci_note"])

            # --------------------------------------------------
            # VISUALIZATION
            # --------------------------------------------------

            try:
                st.altair_chart(_forecast_chart(df), width="stretch")
            except Exception as e:
                st.warning(f"Chart unavailable: {str(e)}")

            # --------------------------------------------------
            # TABLE
            # --------------------------------------------------

            display_df = df.copy().reset_index(drop=True)
            display_df.index = display_df.index + 1
            display_df.index.name = "Observation"

            st.dataframe(display_df, width="stretch")

            # --------------------------------------------------
            # METRICS
            # --------------------------------------------------

            if metrics:
                st.markdown("**Supporting Accuracy Metrics**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE", f"{metrics.get('MAE', 0):,.2f}")
                c2.metric("RMSE", f"{metrics.get('RMSE', 0):,.2f}")
                c3.metric("MAPE", f"{metrics.get('MAPE', 0):,.2f}%")
                c4.metric("Bias", f"{metrics.get('Bias', 0):,.2f}")

            if diagnostics.get("notes"):
                st.caption(diagnostics["notes"])
