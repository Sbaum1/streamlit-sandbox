# FILE: analysis/executive_narratives.py
# ROLE: EXECUTIVE NARRATIVE GENERATOR (READ-ONLY)
# STATUS: CANONICAL / EXECUTIVE-GRADE
# ==================================================
#
# PURPOSE:
#   Translate EXECUTIVE ASSESSMENT + diagnostics into
#   defensible, decision-ready executive narratives.
#
# CORE RULE (HARD):
#   - This module NEVER determines confidence
#   - Confidence posture is ACCEPTED, not computed
#
# GOVERNANCE:
#   - Deterministic
#   - No model ranking
#   - No ensembling
#   - Advisory, not prescriptive
# ==================================================

from __future__ import annotations
from typing import Dict, Any


# --------------------------------------------------
# FORMATTERS (SAFE)
# --------------------------------------------------

def _fmt_pct(x: float | None) -> str:
    if x is None or x != x:
        return "N/A"
    return f"{x:.1%}"


def _fmt_num(x: float | None) -> str:
    if x is None or x != x:
        return "N/A"
    return f"{x:,.0f}"


# --------------------------------------------------
# DECISION USE GUIDANCE (EXPLANATORY ONLY)
# --------------------------------------------------

def _decision_guidance(confidence: str) -> str:
    if confidence == "High":
        return (
            "Suitable for near-term operational planning and executive decision-making, "
            "including budgeting, capacity, and demand alignment."
        )
    if confidence == "Moderate":
        return (
            "Appropriate for directional planning and scenario comparison. "
            "Use alongside conservative assumptions."
        )
    return (
        "Use cautiously. Intended for exploratory analysis, stress-testing, "
        "or as a secondary reference only."
    )


# --------------------------------------------------
# EXECUTIVE NARRATIVE GENERATOR
# --------------------------------------------------

def generate_executive_narrative(
    model_name: str,
    metrics: Dict[str, float],
    diagnostics: Dict[str, Any],
    executive_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generates an executive-grade narrative for a single forecast model.

    REQUIRED INPUT:
    - executive_assessment (from forecast_runner)

    Output structure:
    {
        "confidence": str,
        "summary": str,
        "insights": list[str],
        "decision_guidance": str,
        "risk_flags": list[str]
    }
    """

    # --------------------------------------------------
    # AUTHORITATIVE EXECUTIVE SIGNALS (SOURCE OF TRUTH)
    # --------------------------------------------------
    confidence = executive_assessment.get("confidence_posture", "Low")
    decision_use = executive_assessment.get("decision_use", "Exploratory only")
    risk_flags = list(executive_assessment.get("risk_flags", []))

    mae = metrics.get("MAE")
    rmse = metrics.get("RMSE")
    mape = metrics.get("MAPE")

    bias_dir = diagnostics.get("bias_direction")
    coverage = diagnostics.get("ci_coverage")
    regime = diagnostics.get("regime_flag")

    insights: list[str] = []

    # --------------------------------------------------
    # ACCURACY CONTEXT (FACTUAL)
    # --------------------------------------------------
    if mape is not None:
        insights.append(
            f"Observed forecast accuracy reflects an average error of "
            f"{_fmt_pct(mape)} (MAE {_fmt_num(mae)}, RMSE {_fmt_num(rmse)})."
        )
    else:
        insights.append("Forecast accuracy could not be reliably evaluated.")

    # --------------------------------------------------
    # BIAS INTERPRETATION
    # --------------------------------------------------
    if bias_dir == "Over-forecasting":
        insights.append("Model exhibits a tendency to overstate demand.")
    elif bias_dir == "Under-forecasting":
        insights.append("Model exhibits a tendency to understate demand.")
    else:
        insights.append("No material systematic forecast bias detected.")

    # --------------------------------------------------
    # UNCERTAINTY & INTERVAL CONTEXT
    # --------------------------------------------------
    if coverage is None:
        insights.append("Confidence interval calibration is unavailable or inconclusive.")
    else:
        if coverage < 0.6:
            insights.append("Confidence intervals appear too narrow relative to outcomes.")
        elif coverage > 0.95:
            insights.append("Confidence intervals appear conservative.")
        else:
            insights.append("Confidence intervals are reasonably calibrated.")

    # --------------------------------------------------
    # REGIME SIGNAL
    # --------------------------------------------------
    if regime == "Possible structural change":
        insights.append(
            "Recent error behavior differs from historical patterns, "
            "indicating potential structural instability."
        )
    elif regime:
        insights.append("No material evidence of recent structural instability detected.")

    # --------------------------------------------------
    # EXECUTIVE SUMMARY (NON-NEGOTIABLE WORDING)
    # --------------------------------------------------
    summary = (
        f"{model_name} is assessed as **{confidence} confidence** for decision use. "
        f"This posture is based on observed accuracy, uncertainty calibration, "
        f"and structural diagnostics."
    )

    return {
        "confidence": confidence,
        "summary": summary,
        "insights": insights,
        "decision_guidance": _decision_guidance(confidence),
        "risk_flags": risk_flags,
    }

