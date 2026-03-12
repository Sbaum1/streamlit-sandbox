# ==================================================
# FILE: forecastiq/tabs/report_builder.py
# VERSION: 2.0.0
# ROLE: REPORT BUILDER / EXPORT STUDIO
# UPDATED: Phase 4 — Sentinel engine metadata,
#          certification results, model scores
#          added to Excel export package.
# ==================================================

import streamlit as st
import pandas as pd
import tempfile
import os
from utils.exports import export_excel


def render_report_builder():
    st.header("Report Builder / Export Studio")

    if st.session_state.committed_df is None:
        st.info("Commit data to enable exports.")
        return

    # =========================================================================
    # BUILD EXPORT PACKAGE
    # =========================================================================
    package = {}

    # ── Core data ─────────────────────────────────────────────────────────────
    package["Committed_Data"] = st.session_state.committed_df

    # ── Model metrics ─────────────────────────────────────────────────────────
    if st.session_state.latest_metrics is not None:
        package["Model_Metrics"] = st.session_state.latest_metrics

    # ── Certification metadata ────────────────────────────────────────────────
    cert = st.session_state.sentinel_cert_metadata
    if cert:
        cert_df = pd.DataFrame(cert)
        # Format MASE
        if "MASE" in cert_df.columns:
            cert_df["MASE"] = cert_df["MASE"].apply(
                lambda v: f"{v:.4f}" if v is not None and not (
                    isinstance(v, float) and pd.isna(v)) else "—"
            )
        package["Certification_Results"] = cert_df

    # ── Primary Ensemble forecast ─────────────────────────────────────────────
    if st.session_state.sentinel_primary_df is not None:
        pe = st.session_state.sentinel_primary_df
        package["Primary_Ensemble_Forecast"] = pd.DataFrame({
            "date":     pe.index,
            "forecast": pe.values,
        })

    # ── Stacked Ensemble forecast ─────────────────────────────────────────────
    if st.session_state.sentinel_stacked_df is not None:
        se = st.session_state.sentinel_stacked_df
        package["Stacked_Ensemble_Forecast"] = pd.DataFrame({
            "date":     se.index,
            "forecast": se.values,
        })

    # ── All model forecasts ───────────────────────────────────────────────────
    forecasts = st.session_state.latest_forecasts
    if forecasts:
        all_rows = []
        for model, series in forecasts.items():
            for date, val in zip(series.index, series.values):
                all_rows.append({"model": model, "date": date, "forecast": val})
        if all_rows:
            package["All_Model_Forecasts"] = pd.DataFrame(all_rows)

    # ── Scenario overlay ──────────────────────────────────────────────────────
    if st.session_state.scenario_state.get("enabled") \
            and st.session_state.scenario_forecast_df is not None:
        sc = st.session_state.scenario_forecast_df
        package["Scenario_Overlay"] = pd.DataFrame({
            "date":              sc.index,
            "scenario_forecast": sc.values,
        })

    # ── Run signature ─────────────────────────────────────────────────────────
    if st.session_state.run_signature is not None:
        package["Run_Signature"] = pd.DataFrame(
            list(st.session_state.run_signature.items()),
            columns=["key", "value"],
        )

    # ── Engine metadata ───────────────────────────────────────────────────────
    run_meta = st.session_state.sentinel_run_metadata
    if run_meta:
        meta_rows = []
        for k, v in run_meta.items():
            if not isinstance(v, (dict, list)):
                meta_rows.append({"key": k, "value": str(v)})
        if meta_rows:
            package["Engine_Metadata"] = pd.DataFrame(meta_rows)

    # ── Audit log ─────────────────────────────────────────────────────────────
    if st.session_state.audit_log:
        package["Audit_Log"] = pd.DataFrame(st.session_state.audit_log)

    # =========================================================================
    # EXPORT SUMMARY
    # =========================================================================
    st.subheader("Export Package Contents")

    for sheet_name in package:
        df = package[sheet_name]
        if isinstance(df, pd.DataFrame):
            st.caption(f"✅  {sheet_name} — {len(df)} rows")

    st.divider()

    # =========================================================================
    # EXPORT OPTIONS
    # =========================================================================
    st.subheader("Export Options")

    # CSV — committed data
    st.download_button(
        label="Download Committed Data (CSV)",
        data=st.session_state.committed_df.to_csv(index=False),
        file_name="forecastiq_committed_data.csv",
        mime="text/csv",
    )

    # CSV — certification results
    if cert:
        st.download_button(
            label="Download Certification Results (CSV)",
            data=pd.DataFrame(cert).to_csv(index=False),
            file_name="sentinel_certification_results.csv",
            mime="text/csv",
        )

    # Excel — full workbook
    if st.button("Generate Full Excel Workbook"):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ForecastIQ_Sentinel_Report.xlsx")
            export_excel(package, path)
            with open(path, "rb") as f:
                st.download_button(
                    label="⬇️  Download Excel Workbook",
                    data=f.read(),
                    file_name="ForecastIQ_Sentinel_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    st.caption(
        "Excel workbook includes: committed data, model metrics, certification results, "
        "Primary Ensemble forecast, Stacked Ensemble forecast, all model forecasts, "
        "scenario overlay (if active), run signature, engine metadata, and full audit log."
    )
