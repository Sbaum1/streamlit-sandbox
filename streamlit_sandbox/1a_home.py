# ==================================================
# FILE: home.py
# ROLE: EXECUTIVE HOME SURFACE / REGIME-AWARE
# STATUS: EXECUTION MODES + RISK + CERTIFICATION-AWARE
# ==================================================

import streamlit as st
import numpy as np
import pandas as pd

from sidebar import render_sidebar

from intake.parser import parse_text_data, parse_file
from intake.validator import validate
from intake.fingerprint import fingerprint_df

from analysis.summary import summarize
from analysis.yoy_visuals import yoy_chart

from analysis.forecast_runner import run_all_models
from certification.primary_ensemble import run_primary_ensemble
from analysis.forecast_tables import render_forecast_tables

from state.governance import lock_after_commit, reset_all

CERTIFICATION_MIN_OBS = 36


# ==================================================
# REGIME SHIFT DETECTOR
# ==================================================
def detect_regime_shift(df: pd.DataFrame) -> dict:

    series = df["value"].astype(float)

    if len(series) < 24:
        return {"regime_shift": False}

    returns = series.pct_change().dropna()

    baseline_vol = returns.std()
    recent_vol = returns.rolling(12).std().iloc[-1]

    if baseline_vol == 0 or np.isnan(recent_vol):
        return {"regime_shift": False}

    ratio = recent_vol / baseline_vol

    return {
        "regime_shift": ratio > 1.75,
        "baseline_vol": baseline_vol,
        "recent_vol": recent_vol,
        "vol_ratio": ratio,
    }


# ==================================================
# DISAGREEMENT INDICATOR (UNCHANGED)
# ==================================================
def render_disagreement_indicator(results: dict):

    if not isinstance(results, dict):
        return

    forecasts = []
    model_names = []
    last_observed = None

    for r in results.values():

        if not isinstance(r, dict):
            continue
        if r.get("status") != "success":
            continue
        if r.get("metadata", {}).get("diagnostic_only", False):
            continue

        df = r.get("forecast_df")
        if df is None or df.empty:
            continue

        hist_rows = df[df["ci_low"].isna()]
        if not hist_rows.empty:
            last_observed = pd.to_datetime(hist_rows["date"]).max()
            break

    if last_observed is None:
        return

    horizon = None

    for name, r in results.items():

        if not isinstance(r, dict):
            continue
        if r.get("status") != "success":
            continue
        if r.get("metadata", {}).get("diagnostic_only", False):
            continue

        df = r.get("forecast_df")
        if df is None or df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        future_block = df[df["date"] > last_observed]

        if future_block.empty:
            continue

        future_values = future_block["forecast"].values

        if not np.isfinite(future_values).all():
            continue

        if horizon is None:
            horizon = len(future_values)

        if len(future_values) != horizon:
            continue

        forecasts.append(future_values)
        model_names.append(name)

    if len(forecasts) < 2:
        return

    matrix = np.vstack(forecasts)

    disagreement = np.std(matrix, axis=0)
    avg_disagreement = float(np.mean(disagreement))
    avg_level = float(np.mean(matrix))

    st.markdown("### Model Disagreement Indicator")

    col1, col2 = st.columns(2)
    col1.metric("Models Compared", len(model_names))
    col2.metric("Avg Disagreement", f"{avg_disagreement:,.2f}")

    if avg_level == 0:
        st.info("Insufficient scale to evaluate disagreement.")
        return

    ratio = avg_disagreement / avg_level

    if ratio < 0.05:
        st.success("Low disagreement — strong directional alignment.")
    elif ratio < 0.15:
        st.warning("Moderate disagreement — monitor structural risk.")
    else:
        st.error("High disagreement — elevated forecast uncertainty.")


# ==================================================
# MAIN RENDER FUNCTION
# ==================================================
def render():

    with st.sidebar:
        st.title("Executive Controls")
        render_sidebar(
            disabled=st.session_state.get("dataset_committed", False)
        )

    st.title("Executive Sales Forecasting & Decision Intelligence")
    st.caption("Deterministic, auditable forecasting context")

    if "execution_mode" not in st.session_state:
        st.session_state.execution_mode = "Executive Default Mode"

    execution_mode = st.radio(
        "Forecast Execution Mode",
        ["Executive Default Mode", "Portfolio Mode"],
        index=0 if st.session_state.execution_mode == "Executive Default Mode" else 1,
        horizontal=True,
    )

    st.session_state.execution_mode = execution_mode

    # --------------------------------------------------
    # DATA INTAKE
    # --------------------------------------------------
    st.subheader("Data Intake")

    method = st.radio(
        "Data Input Method",
        ["Paste Data", "Upload File"],
        disabled=st.session_state.get("dataset_committed", False),
    )

    df = None

    if method == "Paste Data":
        text = st.text_area(
            "Paste sales data (date,value or date value)",
            height=220,
            disabled=st.session_state.get("dataset_committed", False),
        )
        if text and text.strip():
            try:
                df = validate(parse_text_data(text))
            except Exception as e:
                st.error(f"Data parsing failed: {e}")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
            disabled=st.session_state.get("dataset_committed", False),
        )
        if uploaded_file is not None:
            try:
                df = validate(parse_file(uploaded_file))
            except Exception as e:
                st.error(f"File parsing failed: {e}")

    # --------------------------------------------------
    # PRE-COMMIT
    # --------------------------------------------------
    if df is not None and not st.session_state.get("dataset_committed", False):

        st.altair_chart(yoy_chart(df), width="stretch")

        if st.button("Commit Dataset"):
            st.session_state.dataset_df = df
            st.session_state.dataset_fingerprint = fingerprint_df(df)
            lock_after_commit()
            st.success("Dataset committed successfully.")

    # --------------------------------------------------
    # POST-COMMIT
    # --------------------------------------------------
    if st.session_state.get("dataset_committed", False):

        dataset_df = st.session_state.dataset_df
        summary = summarize(dataset_df)

        c1, c2 = st.columns(2)
        c1.metric("Observations", summary["observations"])
        c1.metric("CAGR", f"{summary['cagr']:.2%}")
        c2.metric("Volatility", f"{summary['volatility']:.2%}")
        c2.write(f"{summary['start']} → {summary['end']}")

        # --------------------------------------------------
        # CERTIFICATION ELIGIBILITY NOTICE
        # --------------------------------------------------
        obs_count = summary["observations"]

        if obs_count < CERTIFICATION_MIN_OBS:
            st.warning(
                f"Certification requires minimum {CERTIFICATION_MIN_OBS} monthly observations. "
                f"Current dataset contains {obs_count}. "
                "Metrics and readiness tiers will be unavailable."
            )
        else:
            st.success("Dataset eligible for certification scoring.")

        # --------------------------------------------------
        # REGIME DETECTION
        # --------------------------------------------------
        regime_info = detect_regime_shift(dataset_df)

        auto_stress = False

        if regime_info.get("regime_shift"):
            auto_stress = True
            st.error(
                f"⚠ Regime Shift Detected | Volatility Ratio: {regime_info['vol_ratio']:.2f}"
            )

        manual_stress = st.checkbox("Apply Additional Stress (+15%)")
        stress_enabled = auto_stress or manual_stress

        # --------------------------------------------------
        # RUN FORECAST
        # --------------------------------------------------
        if not st.session_state.get("forecast_completed", False):
            if st.button("Run Forecast"):
                try:

                    if execution_mode == "Executive Default Mode":
                        result = run_primary_ensemble(
                            df=dataset_df,
                            horizon=st.session_state.forecast_horizon,
                            confidence_level=st.session_state.confidence_level,
                        )
                        results = {
                            "Primary Ensemble": {
                                "status": "success",
                                "forecast_df": result.forecast_df,
                                "metadata": result.metadata,
                                "metrics": result.metrics,
                            }
                        }
                    else:
                        results = run_all_models(
                            df=dataset_df,
                            horizon=st.session_state.forecast_horizon,
                            confidence_level=st.session_state.confidence_level,
                        )

                    if stress_enabled:
                        for r in results.values():
                            df_out = r.get("forecast_df")
                            if df_out is not None:
                                width = df_out["ci_high"] - df_out["ci_low"]
                                shock = width * 0.15
                                df_out["ci_low"] -= shock
                                df_out["ci_high"] += shock

                    st.session_state.forecast_results = results
                    st.session_state.forecast_completed = True
                    st.success("Forecast completed.")

                except Exception as e:
                    st.error(f"Forecast execution failed: {e}")

        if st.session_state.get("forecast_completed", False):

            results = st.session_state.forecast_results

            if execution_mode == "Portfolio Mode":
                render_disagreement_indicator(results)

            render_forecast_tables(results)

        st.button("Reset Session", on_click=reset_all)
