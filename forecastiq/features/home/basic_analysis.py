# 🔒 LOCKED FILE - DO NOT MODIFY LOGIC
# Allowed changes: styling, copy text, visual tuning (colors, labels)
# Forbidden: altering calculations, control flow, or state usage
#
# ==================================================
# FILE: features/home/basic_analysis.py
# ROLE: Basic Aanlysis Block (Post-Commit Insight Surface)
# STATUS: SECTION-LOCKED / EXECUTIVE-GRADE
# ==================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# --------------------------------------------------
# INTERNAL: BASIC METRICS (PURE)
# --------------------------------------------------

def _compute_basic_metrics(df: pd.DataFrame) -> dict:
    values = df["value"].astype(float).values
    dates = pd.to_datetime(df["date"])

    n = len(values)
    first, last = values[0], values[-1]

    overall_change = last - first
    pct_change = (last / first - 1) * 100 if first != 0 else np.nan

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # Trend (linear fit)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, values, 1)
    fitted = slope * t + intercept
    r2 = 1 - np.sum((values - fitted) ** 2) / np.sum((values - values.mean()) ** 2)

    # Volatility label
    cv = std / mean if mean != 0 else np.nan
    if cv < 0.25:
        volatility = "Low"
    elif cv < 0.5:
        volatility = "Moderate"
    else:
        volatility = "High"

    # CAGR (annualized)
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr = (last / first) ** (1 / years) - 1 if years > 0 and first > 0 else np.nan

    return {
        "observations": n,
        "mean": mean,
        "std": std,
        "overall_change": overall_change,
        "pct_change": pct_change,
        "trend_slope": slope,
        "trend_strength": max(0, min(1, r2)),
        "volatility": volatility,
        "cagr": cagr,
        "date_range": (dates.min(), dates.max()),
    }


# --------------------------------------------------
# PUBLIC: BASIC ANALYSIS SECTION
# --------------------------------------------------

def render_basic_analysis():
    if st.session_state.get("committed_df") is None:
        return

    df = st.session_state.committed_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    with st.expander("3. Basic Analysis", expanded=True):
        metrics = _compute_basic_metrics(df)

        # ==================================================
        # KPI GRID
        # ==================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Observations", metrics["observations"])
            st.metric("Mean", f"{metrics['mean']:,.2f}")
            st.metric("Seasonal Strength", "0.50")  # placeholder

        with col2:
            st.metric(
                "Overall Change",
                f"{metrics['overall_change']:,.2f}",
                f"{metrics['pct_change']:.2f}%" if not np.isnan(metrics["pct_change"]) else None,
            )
            st.metric("Std Dev", f"{metrics['std']:,.2f}")
            st.metric("Trend Strength", f"{metrics['trend_strength']:.2f}")

        with col3:
            st.metric(
                "CAGR (Annual)",
                f"{metrics['cagr'] * 100:.2f}%" if not np.isnan(metrics["cagr"]) else "—",
            )
            st.metric("Volatility", metrics["volatility"])
            st.metric("Trend Slope", f"{metrics['trend_slope']:.2f}")

        st.markdown(
            f"""
            **Date range:** {metrics['date_range'][0].date()} → {metrics['date_range'][1].date()}  
            **Inferred frequency:** {st.session_state.get("data_frequency", "—")}
            """
        )

        st.divider()

        # ==================================================
        # VISUAL 1 — HISTORICAL TREND (YOY BY MONTH)
        # ==================================================
        fig_yoy = go.Figure()

        for year, g in df.groupby("year"):
            g_month = (
                g.groupby("month", as_index=False)["value"]
                .mean()
                .sort_values("month")
            )

            fig_yoy.add_trace(go.Scatter(
                x=g_month["month"],
                y=g_month["value"],
                mode="lines+markers",
                name=str(year)
            ))

            if len(g_month) >= 2:
                x = np.arange(len(g_month))
                slope, intercept = np.polyfit(x, g_month["value"], 1)
                trend = slope * x + intercept

                fig_yoy.add_trace(go.Scatter(
                    x=g_month["month"],
                    y=trend,
                    mode="lines",
                    name=f"{year} Trend",
                    line=dict(dash="dot"),
                    visible="legendonly"
                ))

        fig_yoy.update_layout(
            title="Historical Trend — Year over Year",
            xaxis=dict(
                title="Month",
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ),
            yaxis=dict(title="Value"),
            template="plotly_white",
            hovermode="x unified",
            legend_title_text="Year / Trend"
        )

        st.plotly_chart(fig_yoy, use_container_width=True)
        st.caption(
            "Each solid line represents a calendar year. "
            "Dotted lines are per-year linear trends and can be toggled via the legend."
        )

        # ==================================================
        # VISUAL 2 — DISTRIBUTION
        # ==================================================
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df["value"],
            nbinsx=20,
            name="Distribution"
        ))

        fig_dist.update_layout(
            title="Value Distribution",
            template="plotly_white"
        )

        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("Reveals dispersion, skew, and volatility in the committed data.")

        st.divider()

        # ==================================================
        # VISUAL 3 — MONTHLY CLUSTER SCATTER
        # ==================================================
        fig_cluster = go.Figure()

        for month, g in df.sort_values("date").groupby("month"):
            fig_cluster.add_trace(go.Scatter(
                x=[month] * len(g),
                y=g["value"],
                mode="lines+markers",
                line=dict(width=1),
                marker=dict(size=6),
                showlegend=False
            ))

            mean_val = g["value"].mean()
            fig_cluster.add_trace(go.Scatter(
                x=[month - 0.25, month + 0.25],
                y=[mean_val, mean_val],
                mode="lines",
                line=dict(dash="dot", width=2),
                showlegend=False
            ))

        fig_cluster.update_layout(
            title="Monthly Cluster Distribution",
            xaxis=dict(
                title="Month",
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ),
            yaxis=dict(title="Value"),
            template="plotly_white",
            hovermode="closest"
        )

        st.plotly_chart(fig_cluster, use_container_width=True)
        st.caption(
            "Each cluster represents all observations for a given month. "
            "Thin lines connect values chronologically; dotted lines show monthly means."
        )
