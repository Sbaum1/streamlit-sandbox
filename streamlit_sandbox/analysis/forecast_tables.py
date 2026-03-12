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
# - Certification eligibility hint enforced (36 observations minimum)
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
OPTIONAL_COLUMNS = {"actual", "ci_low", "ci_high", "ci_mid", "error_pct"}


# ==================================================
# HELPERS
# ==================================================

def _as_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _safe_num(v, decimals=2):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except Exception:
        return None


def _fmt(v, decimals=2, suffix=""):
    n = _safe_num(v, decimals=decimals)
    if n is None:
        return "N/A"
    return f"{n:,.{decimals}f}{suffix}"


def _metrics_get(metrics: dict, key: str, default=None):
    if not isinstance(metrics, dict):
        return default
    return metrics.get(key, default)


# ==================================================
# READINESS + CONFIDENCE (EXECUTIVE-FACING)
# ==================================================

def _readiness_badge_text(executive_assessment: dict, metrics: dict) -> str:
    if not isinstance(executive_assessment, dict):
        executive_assessment = {}
    if not isinstance(metrics, dict):
        metrics = {}

    tier = executive_assessment.get("readiness_tier")
    if tier:
        return str(tier)

    # Fallback if tier missing
    eligible = metrics.get("eligible")
    if eligible is False:
        obs = metrics.get("Observations") or metrics.get("observations")
        reason = metrics.get("reason", "Minimum data requirement not met.")
        if obs is not None:
            return f"Ineligible (n={obs}) — {reason}"
        return f"Ineligible — {reason}"

    if eligible is True:
        return "Unscored — Certification metrics incomplete"

    return "Not Assessed"


def _render_readiness_panel(badge_text: str, metrics: dict, executive_assessment: dict, metadata: dict):

    st.markdown("### Decision Readiness")
    left, right = st.columns([2, 3])

    left.metric("Readiness Tier", badge_text)
    right.caption(CERTIFICATION_MIN_HINT)

    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(executive_assessment, dict):
        executive_assessment = {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Confidence posture (now provided by forecast_runner.py)
    confidence_posture = executive_assessment.get("confidence_posture", "Not Assessed")
    st.metric("Confidence Posture", confidence_posture)

    # If ineligible, show the reason prominently
    if "Ineligible" in (badge_text or "") or metrics.get("eligible") is False:
        reason = metrics.get("reason")
        obs = metrics.get("Observations") or metrics.get("observations")
        if reason:
            st.warning(f"Certification Ineligible: {reason}")
        if obs is not None:
            st.info(f"Observed history length: {obs}")

    # Risk flags from forecast_runner.py executive_assessment
    risk_flags = executive_assessment.get("risk_flags") or []
    if isinstance(risk_flags, list) and risk_flags:
        st.warning("Risk Flags: " + ", ".join([str(x) for x in risk_flags]))

    # Decision guidance from forecast_runner.py executive_assessment
    guidance = executive_assessment.get("decision_guidance")
    if guidance:
        st.info(str(guidance))

    # CI disclosure (if any)
    if metadata.get("ci_note"):
        st.info(str(metadata["ci_note"]))


# ==================================================
# SAFE CHART BUILDER
# ==================================================

def _forecast_chart(df: pd.DataFrame) -> alt.Chart:

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Forecast output is not a dataframe.")

    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError("Forecast dataframe missing required columns.")

    df = _as_datetime(df)

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
# METRICS PANEL (EXECUTIVE-SCAN FRIENDLY)
# ==================================================

def _render_metrics_panel(metrics: dict, confidence_level: float):

    if not isinstance(metrics, dict) or not metrics:
        st.caption("No certification metrics available.")
        return

    eligible = metrics.get("eligible")
    if eligible is False:
        reason = metrics.get("reason", "Minimum data requirement not met.")
        st.warning(f"Certification Metrics: Ineligible — {reason}")
        return

    # Keys here match forecast_runner.py normalization
    obs = _metrics_get(metrics, "Observations")
    folds = _metrics_get(metrics, "Folds")

    mase = _metrics_get(metrics, "MASE")
    theils_u = _metrics_get(metrics, "Theils_U")
    coverage = _metrics_get(metrics, "CI_Coverage")

    mae = _metrics_get(metrics, "MAE")
    rmse = _metrics_get(metrics, "RMSE")
    mape = _metrics_get(metrics, "MAPE")
    smape = _metrics_get(metrics, "SMAPE")
    bias = _metrics_get(metrics, "Bias")

    st.markdown("### Certification / Backtest Metrics")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Observations", f"{int(obs)}" if _safe_num(obs) is not None else "N/A")
    r2.metric("Folds", f"{int(folds)}" if _safe_num(folds) is not None else "N/A")
    r3.metric("MASE", _fmt(mase, decimals=3))
    r4.metric("Theil's U", _fmt(theils_u, decimals=3))

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("MAE", _fmt(mae, decimals=2))
    r6.metric("RMSE", _fmt(rmse, decimals=2))
    r7.metric("MAPE", _fmt(mape, decimals=2, suffix="%"))
    r8.metric("Bias", _fmt(bias, decimals=3))

    # Coverage callout
    if _safe_num(coverage) is not None:
        cov = float(coverage)
        st.metric("CI Coverage", _fmt(cov * 100.0, decimals=1, suffix="%"))
        # Flag if badly miscalibrated
        if abs(cov - confidence_level) > 0.10:
            st.warning(
                f"CI calibration warning: observed coverage {_fmt(cov*100,1,'%')} "
                f"vs target {_fmt(confidence_level*100,1,'%')}."
            )


# ==================================================
# MAIN RENDERER
# ==================================================

def render_forecast_tables(results: dict):

    if not isinstance(results, dict):
        st.error("Forecast results structure invalid.")
        return

    st.subheader("Forecast Results by Model")
    st.caption(MODEL_RANKING_DISCLOSURE)
    st.caption(CERTIFICATION_MIN_HINT)

    failures = results.get("_failures", [])
    if isinstance(failures, list):
        _render_failure_panel(failures)

    # Pull confidence level if present anywhere (fallback to 0.8)
    confidence_level = 0.8
    for payload in results.values():
        if isinstance(payload, dict):
            md = payload.get("metadata") or {}
            cl = md.get("confidence_level")
            if cl is not None:
                try:
                    confidence_level = float(cl)
                    break
                except Exception:
                    pass

    for model_name, payload in results.items():

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

            df = _as_datetime(df)

            # --------------------------------------------------
            # READINESS PANEL (EXECUTIVE-FIRST)
            # --------------------------------------------------
            _render_readiness_panel(
                badge_text=badge_text,
                metrics=metrics,
                executive_assessment=executive_assessment,
                metadata=metadata,
            )

            # --------------------------------------------------
            # EXECUTIVE NARRATIVE (OPTIONAL / FAILURE-SAFE)
            # --------------------------------------------------
            st.markdown("### Executive Assessment Narrative")

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
            # CHART
            # --------------------------------------------------
            try:
                st.altair_chart(_forecast_chart(df), width="stretch")
            except Exception as e:
                st.warning(f"Chart unavailable: {str(e)}")

            # --------------------------------------------------
            # TABLE (EXECUTIVE VIEW)
            # --------------------------------------------------
            st.markdown("### Forecast Table")

            display_df = df.copy()

            # Ensure optional columns exist so PDF/table doesn’t look “broken”
            for col in OPTIONAL_COLUMNS:
                if col not in display_df.columns:
                    display_df[col] = pd.NA

            # Preferred column order (stable, executive-grade)
            ordered_cols = ["date", "actual", "forecast", "ci_low", "ci_mid", "ci_high", "error_pct"]
            display_df = display_df[[c for c in ordered_cols if c in display_df.columns]]

            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
            display_df.index.name = "Observation"

            st.dataframe(display_df, width="stretch")

            # --------------------------------------------------
            # METRICS (NOW MATCHES forecast_runner.py)
            # --------------------------------------------------
            _render_metrics_panel(metrics=metrics, confidence_level=confidence_level)

            # Diagnostics notes (if any)
            if isinstance(diagnostics, dict) and diagnostics.get("notes"):
                st.caption(diagnostics["notes"])
