# ==================================================
# FILE: forecastiq/utils/visuals.py
# ROLE: EXECUTIVE VISUAL SYSTEM (CANONICAL)
# STATUS: REGRESSION-SAFE / SCENARIO-READY
# ==================================================

from __future__ import annotations
from typing import Dict


# ==================================================
# BASE LAYOUT (RESERVED)
# ==================================================

def base_layout(title: str, subtitle: str) -> Dict:
    """
    Reserved for future global layout theming.
    Intentionally minimal to avoid regressions.
    """
    return {}


# ==================================================
# EXECUTIVE COLOR PALETTE
# ==================================================

EXECUTIVE_COLORS = {
    "actual": "#1F2937",        # Slate / historical truth
    "forecast": "#C8974A",      # Executive blue
    "comparison": "#9CA3AF",    # Muted gray
    "upside": "#16A34A",        # Green (growth)
    "downside": "#C45858",      # Red (risk)
    "band": "rgba(200,151,74,0.15)",
}


# ==================================================
# LINE STYLES — SEMANTIC, NOT DECORATIVE
# ==================================================

def executive_line_style(kind: str) -> Dict:
    """
    Canonical line styles used across all executive charts.
    """

    palette = {
        "actual": dict(color=EXECUTIVE_COLORS["actual"], width=2),
        "forecast": dict(color=EXECUTIVE_COLORS["forecast"], width=3),
        "comparison": dict(color=EXECUTIVE_COLORS["comparison"], width=1, dash="dot"),
        "upside": dict(color=EXECUTIVE_COLORS["upside"], width=2, dash="dash"),
        "downside": dict(color=EXECUTIVE_COLORS["downside"], width=2, dash="dash"),
    }

    return dict(
        mode="lines",
        line=palette.get(kind, {}),
        hovertemplate="%{y}<extra></extra>",
    )


# ==================================================
# CONFIDENCE BAND STYLES (NO GEOMETRY)
# ==================================================

def confidence_band_style(kind: str) -> Dict:
    """
    Visual contract for uncertainty bands.

    IMPORTANT:
    - This function NEVER sets `fill`
    - Geometry (fill='tonexty') is applied ONLY in chart code
    """

    if kind == "upper":
        return dict(
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )

    return dict(
        line=dict(width=0),
        fillcolor=EXECUTIVE_COLORS["band"],
        showlegend=False,
        hoverinfo="skip",
    )


# ==================================================
# SCENARIO LABELS
# ==================================================

SCENARIO_LABELS = {
    "base": "Baseline Forecast",
    "upside": "Upside Scenario",
    "downside": "Downside Scenario",
}


# ==================================================
# TABLE HELPERS
# ==================================================

def format_forecast_table(df):
    """
    Applies executive-safe formatting rules to forecast tables.
    Does NOT mutate input.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if "date" in out.columns:
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    for c in out.select_dtypes("number").columns:
        out[c] = out[c].round(2)

    return out


# ==================================================
# EXPLAINABILITY HELPERS
# ==================================================

def explainability_badge(label: str, value: str) -> str:
    return f"**{label}:** {value}"
