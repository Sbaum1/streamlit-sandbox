# FILE: analysis/summary.py
# ROLE: EXECUTIVE DATASET SUMMARY & CONTEXT
# STATUS: CANONICAL / EXECUTIVE-GRADE
# ==================================================

import pandas as pd
import numpy as np


def summarize(df: pd.DataFrame) -> dict:
    """
    Generates an executive-grade descriptive summary of the dataset.

    Governance:
    - Deterministic
    - No forecasting
    - No model judgment
    - Transparent metrics only
    """

    df = df.copy().sort_values("date")

    obs = len(df)

    start_date = df["date"].min()
    end_date = df["date"].max()

    years = (end_date - start_date).days / 365.25

    cagr = (
        (df["value"].iloc[-1] / df["value"].iloc[0]) ** (1 / years) - 1
        if years > 0 and df["value"].iloc[0] > 0
        else np.nan
    )

    returns = df["value"].pct_change()
    volatility = returns.std()

    # --------------------------------------------------
    # RECENCY CONTEXT (LAST 12 OBS VS PRIOR)
    # --------------------------------------------------
    recent_window = min(12, obs // 2)

    recent_values = df["value"].iloc[-recent_window:]
    prior_values = df["value"].iloc[-2 * recent_window : -recent_window]

    recent_trend = (
        (recent_values.iloc[-1] / recent_values.iloc[0]) - 1
        if len(recent_values) > 1
        else np.nan
    )

    prior_trend = (
        (prior_values.iloc[-1] / prior_values.iloc[0]) - 1
        if len(prior_values) > 1
        else np.nan
    )

    momentum = (
        "Accelerating"
        if recent_trend > prior_trend
        else "Decelerating"
        if recent_trend < prior_trend
        else "Stable"
    )

    # --------------------------------------------------
    # EXECUTIVE READINESS SIGNAL
    # --------------------------------------------------
    cadence = pd.infer_freq(df["date"])

    readiness = "Decision-Grade"
    if obs < 24:
        readiness = "Limited history"
    if cadence is None:
        readiness = "Irregular cadence"

    return {
        "observations": obs,
        "start": start_date.date(),
        "end": end_date.date(),
        "years_covered": round(years, 2),
        "cagr": cagr,
        "volatility": volatility,
        "recent_trend": recent_trend,
        "prior_trend": prior_trend,
        "momentum": momentum,
        "cadence": cadence,
        "data_readiness": readiness,
    }

