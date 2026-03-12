# ==================================================
# FILE: forecastiq/utils/frequency.py
# VERSION: 2.0.0
# ROLE: FREQUENCY INFERENCE — ROBUST MULTI-STRATEGY
# ==================================================

import pandas as pd


def infer_frequency(dates: pd.Series) -> dict:
    """
    Infer time-series frequency from a date column.
    Handles first-of-month dates, irregular spacing, and edge cases.

    Returns:
        dict with keys: frequency, confidence, details
    """
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()

    if len(dates) < 3:
        return {"frequency": "Unknown", "confidence": 0.0, "details": "Insufficient data"}

    diffs = dates.diff().dropna()
    median_days = diffs.median().days
    std_days    = diffs.std().days if len(diffs) > 1 else 0

    # ── Strategy 1: Try pandas inferred freq on DatetimeIndex ────────────────
    try:
        idx = pd.DatetimeIndex(dates)
        inferred = pd.infer_freq(idx)
        if inferred:
            if inferred in ("MS", "ME", "M", "BMS", "BM"):
                return {"frequency": "Monthly", "confidence": 0.97,
                        "details": f"pandas inferred: {inferred}"}
            if inferred in ("W", "W-MON", "W-SUN", "W-FRI"):
                return {"frequency": "Weekly", "confidence": 0.95,
                        "details": f"pandas inferred: {inferred}"}
            if inferred in ("D", "B"):
                return {"frequency": "Daily", "confidence": 0.97,
                        "details": f"pandas inferred: {inferred}"}
            if inferred in ("QS", "QE", "Q", "BQS", "BQE"):
                return {"frequency": "Quarterly", "confidence": 0.95,
                        "details": f"pandas inferred: {inferred}"}
            if inferred in ("YS", "YE", "Y", "A"):
                return {"frequency": "Annual", "confidence": 0.95,
                        "details": f"pandas inferred: {inferred}"}
    except Exception:
        pass

    # ── Strategy 2: Month-start detection ────────────────────────────────────
    # Handles cases where dates are first-of-month but with irregular day counts
    day_of_month = dates.dt.day
    if (day_of_month == 1).mean() > 0.85:
        # All or nearly all dates fall on the 1st → monthly
        return {"frequency": "Monthly", "confidence": 0.94,
                "details": "First-of-month pattern detected"}

    # ── Strategy 3: Median diff bucketing ────────────────────────────────────
    cv = std_days / median_days if median_days > 0 else 1.0  # coefficient of variation

    if 27 <= median_days <= 33:
        conf = max(0.7, 0.95 - cv * 0.5)
        return {"frequency": "Monthly", "confidence": round(conf, 2),
                "details": f"Median diff {median_days}d (CV={cv:.2f})"}

    if 85 <= median_days <= 95:
        conf = max(0.6, 0.90 - cv * 0.5)
        return {"frequency": "Quarterly", "confidence": round(conf, 2),
                "details": f"Median diff {median_days}d"}

    if 350 <= median_days <= 380:
        return {"frequency": "Annual", "confidence": 0.85,
                "details": f"Median diff {median_days}d"}

    if 13 <= median_days <= 17:
        conf = max(0.6, 0.88 - cv * 0.4)
        return {"frequency": "Bi-Weekly", "confidence": round(conf, 2),
                "details": f"Median diff {median_days}d"}

    if 6 <= median_days <= 8:
        conf = max(0.7, 0.92 - cv * 0.4)
        return {"frequency": "Weekly", "confidence": round(conf, 2),
                "details": f"Median diff {median_days}d"}

    if median_days == 1:
        return {"frequency": "Daily", "confidence": 0.95,
                "details": "Exact 1-day spacing"}

    if 2 <= median_days <= 5:
        return {"frequency": "Daily", "confidence": 0.75,
                "details": f"Near-daily spacing (median {median_days}d)"}

    # ── Strategy 4: Month transition check ───────────────────────────────────
    # If most consecutive dates are in adjacent months
    months = dates.dt.to_period("M")
    month_diffs = months.diff().dropna().apply(lambda x: x.n)
    if (month_diffs == 1).mean() > 0.75:
        return {"frequency": "Monthly", "confidence": 0.80,
                "details": "Month-transition pattern (irregular day spacing)"}

    return {"frequency": "Unknown", "confidence": 0.3,
            "details": f"Median diff {median_days}d — could not classify"}
