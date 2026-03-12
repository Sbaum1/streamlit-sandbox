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
    "No model ranking is performed. Readiness tiers reflect certification eligibility and backtest strength only."
)

CERTIFICATION_MIN_OBS = 36

REQUIRED_COLUMNS = {"date", "forecast"}


# ==================================================
# SAFE CHART BUILDER
# ==================================================

def _forecast_chart(df: pd.DataFrame) -> alt.Chart:

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Forecast output is not a dataframe.")

    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError("Forecast dataframe missing required columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date"),
    )

    layers = []

    if {"ci_low", "ci_high"}.issubset(df.columns):
        if df[["ci_low", "ci_high"]].notna().any().any():
            layers.append(
                base.mark_area(opacity=0.2, color="#d62728")
                .encode(y="ci_low:Q", y2="ci_high:Q")
            )

    if "actual" in df.columns and df["actual"].notna().any():
        layers.append(
            base.mark_line(color="#1f77b4", strokeWidth=2)
            .encode(y="actual:Q")
        )

    layers.append(
        base.mark_line(color="#d62728", strokeWidth=2)
        .encode(y="forecast:Q")
    )

    return alt.layer(*layers).properties(height=280)


# ==================================================
# READINESS PANEL
# ==================================================

def _render_readiness_panel(model_name: str, executive_assessment: dict, metrics: dict, df: pd.DataFrame):

    readiness = executive_assessment.get("readiness_tier", "Not Assessed")
    hist_obs = df["actual"].notna().sum() if "actual" in df.columns else 0

    st.markdown("### Decision Readiness")

    left, right = st.columns([2, 3])
    left.metric("Readiness Tier", readiness)
    right.caption(
        f"Certification requires minimum {CERTIFICATION_MIN_OBS} monthly observations."
    )

    # Explicit Ineligible handling
    if readiness == "Ineligible":
        reason = metrics.get("reason", "Minimum observation threshold not met.")
        st.error(f"Ineligible: {reason}")
        st.info(f"Observed history length: {hist_obs}")
        return

    if readiness == "Unscored":
        st.warning("Model eligible but certification metrics unavailable.")
        return

    if readiness.startswith("Tier 1"):
        st.success("Tier 1 — Production Ready")
    elif readiness.startswith("Tier 2"):
        st.warning("Tier 2 — Acceptable but monitor performance")
    elif readiness.startswith("Tier 3"):
        st.error("Tier 3 — Weak performance, caution advised")
    else:
        st.info("Readiness not assessed.")


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
        if not isinstance(record, dict):
            continue

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

        if isinstance(model_name, str) and model_name.startswith("_"):
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
            # READINESS PANEL (EXECUTIVE-FIRST)
            # --------------------------------------------------

            _render_readiness_panel(
                model_name=model_name,
                executive_assessment=executive_assessment,
                metrics=metrics,
                df=df,
            )

            # --------------------------------------------------
            # EXECUTIVE NARRATIVE
            # --------------------------------------------------

            st.markdown(f"### Executive Assessment — {model_name}")

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
            # CERTIFICATION METRICS (ONLY IF ELIGIBLE)
            # --------------------------------------------------

            if isinstance(metrics, dict) and metrics:

                if metrics.get("eligible") is False:
                    st.warning(metrics.get("reason", "Model not eligible for certification."))
                else:
                    st.markdown("**Certification / Backtest Metrics**")

                    c1, c2, c3, c4 = st.columns(4)

                    mase = metrics.get("mase")
                    theils_u = metrics.get("theils_u")
                    smape = metrics.get("smape")
                    bias = metrics.get("bias")

                    c1.metric("MASE", f"{mase:,.3f}" if mase is not None else "N/A")
                    c2.metric("Theil's U", f"{theils_u:,.3f}" if theils_u is not None else "N/A")
                    c3.metric("sMAPE", f"{smape:,.2f}%" if smape is not None else "N/A")
                    c4.metric("Bias", f"{bias:,.3f}" if bias is not None else "N/A")

                    if metrics.get("notes"):
                        st.caption(str(metrics["notes"]))

            if isinstance(diagnostics, dict) and diagnostics.get("notes"):
                st.caption(diagnostics["notes"])
