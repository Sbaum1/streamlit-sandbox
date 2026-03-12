# FILE: analysis/forecast_tables.py
# ROLE: EXECUTIVE FORECAST RESULT RENDERER (FORTUNE 100 HARDENED + FAILURE SAFE)
# ==================================================
# GOVERNANCE:
# - Safe against non-model entries (e.g., _failures list)
# - Schema validation enforced
# - No ranking or auto-selection
# - Failure panel rendered explicitly
# - Defensive rendering at every boundary
# ==================================================

import streamlit as st
import pandas as pd
import altair as alt

from analysis.executive_narratives import generate_executive_narrative


MODEL_RANKING_DISCLOSURE = (
    "No model ranking is performed. Readiness tiers reflect certification eligibility and backtest strength only."
)

CERTIFICATION_MIN_HINT = (
    "Certification requires a minimum of 36 observations. "
    "Below 36, models may still run, but readiness is marked Ineligible."
)

REQUIRED_COLUMNS = {"date", "forecast"}
OPTIONAL_COLUMNS = {"actual", "ci_low", "ci_high", "error_pct"}


# ==================================================
# READINESS BADGE (EXECUTIVE-FACING)
# ==================================================

def _readiness_badge_text(executive_assessment: dict, metrics: dict) -> str:
    """
    Returns a short executive badge string.
    Expected:
      executive_assessment["readiness_tier"] from forecast_runner.py
      metrics may include: eligible, reason, observations
    """
    if not isinstance(executive_assessment, dict):
        executive_assessment = {}

    tier = executive_assessment.get("readiness_tier")

    # If tier is not present, derive minimally from metrics
    if not tier and isinstance(metrics, dict):
        eligible = metrics.get("eligible")
        if eligible is False:
            obs = metrics.get("observations")
            reason = metrics.get("reason", "Minimum data requirement not met.")
            if obs is not None:
                return f"Ineligible (n={obs}) — {reason}"
            return f"Ineligible — {reason}"
        if eligible is True:
            return "Unscored — Certification metrics incomplete"
        return "Not Assessed"

    return tier or "Not Assessed"


def _render_readiness_panel(model_name: str, badge_text: str, metrics: dict, metadata: dict):
    """
    Visually emphasizes what an executive should consider.
    """
    st.markdown("### Decision Readiness")

    left, right = st.columns([2, 3])

    # Badge + meaning
    left.metric("Readiness Tier", badge_text)

    # Certification hint always visible
    right.caption(CERTIFICATION_MIN_HINT)

    # If ineligible, show the reason prominently
    if "Ineligible" in (badge_text or ""):
        reason = None
        obs = None
        if isinstance(metrics, dict):
            reason = metrics.get("reason")
            obs = metrics.get("observations")
        if reason:
            st.warning(f"Certification Ineligible: {reason}")
        if obs is not None:
            st.info(f"Observed history length: {obs}")

    # CI disclosure (if any)
    if isinstance(metadata, dict) and metadata.get("ci_note"):
        st.info(metadata["ci_note"])


# ==================================================
# SAFE CHART BUILDER
# ==================================================

def _forecast_chart(df: pd.DataFrame) -> alt.Chart:

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Forecast output is not a dataframe.")

    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError("Forecast dataframe missing required columns.")

    # Defensive: ensure date is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

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

    # Executive-wide certification hint (always visible)
    st.caption(CERTIFICATION_MIN_HINT)

    # --------------------------------------------------
    # FAILURE LOG HANDLING (SAFE)
    # --------------------------------------------------

    failures = results.get("_failures", [])
    if isinstance(failures, list):
        _render_failure_panel(failures)

    # --------------------------------------------------
    # MODEL PANELS
    # --------------------------------------------------

    for model_name, payload in results.items():

        # Skip non-model entries (e.g., _failures)
        if not isinstance(payload, dict):
            continue

        # Skip explicit system keys
        if isinstance(model_name, str) and model_name.startswith("_"):
            continue

        status = payload.get("status")
        metrics = payload.get("metrics") or {}
        diagnostics = payload.get("diagnostics") or {}
        executive_assessment = payload.get("executive_assessment") or {}
        metadata = payload.get("metadata") or {}

        badge_text = _readiness_badge_text(executive_assessment, metrics)

        # Put readiness right in the expander label so execs can scan
        expander_label = f"{model_name}  |  {badge_text}"

        with st.expander(expander_label, expanded=False):

            if status != "success":
                st.error(f"Model failed: {payload.get('error')}")
                continue

            df: pd.DataFrame = payload.get("forecast_df")

            if not isinstance(df, pd.DataFrame) or df.empty:
                st.warning("No forecast output produced.")
                continue

            if not REQUIRED_COLUMNS.issubset(df.columns):
                st.error("Forecast output schema invalid.")
                continue

            # --------------------------------------------------
            # READINESS PANEL (EXECUTIVE-FIRST)
            # --------------------------------------------------
            _render_readiness_panel(
                model_name=model_name,
                badge_text=badge_text,
                metrics=metrics,
                metadata=metadata,
            )

            # --------------------------------------------------
            # EXECUTIVE NARRATIVE (OPTIONAL / FAILURE-SAFE)
            # --------------------------------------------------

            st.markdown("### Executive Assessment")

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
            # METRICS (ONLY IF PRESENT + STRUCTURED)
            # --------------------------------------------------

            if isinstance(metrics, dict) and metrics:

                # If ineligible, show reason and avoid fake numbers
                if metrics.get("eligible") is False:
                    reason = metrics.get("reason", "Minimum data requirement not met.")
                    st.warning(f"Certification Metrics: Ineligible — {reason}")
                else:
                    st.markdown("**Certification / Backtest Metrics**")

                    # These keys depend on your backtest_engine output.
                    # We render defensively: only show what exists.
                    cols = st.columns(4)
                    cols[0].metric("MASE", f"{metrics.get('mase', float('nan')):,.3f}" if metrics.get("mase") is not None else "N/A")
                    cols[1].metric("Theil's U", f"{metrics.get('theils_u', float('nan')):,.3f}" if metrics.get("theils_u") is not None else "N/A")
                    cols[2].metric("sMAPE", f"{metrics.get('smape', float('nan')):,.2f}%" if metrics.get("smape") is not None else "N/A")
                    cols[3].metric("Bias", f"{metrics.get('bias', float('nan')):,.3f}" if metrics.get("bias") is not None else "N/A")

                    if metrics.get("notes"):
                        st.caption(str(metrics["notes"]))

            # Diagnostics notes (if any)
            if isinstance(diagnostics, dict) and diagnostics.get("notes"):
                st.caption(diagnostics["notes"])
