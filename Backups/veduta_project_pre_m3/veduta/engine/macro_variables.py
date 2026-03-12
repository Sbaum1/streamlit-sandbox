# ==================================================
# FILE: forecastiq/engine/macro_variables.py
# VERSION: 1.0.0
# ROLE: MACRO VARIABLE INTEGRATION
#       Tier A — FRED API live data fetch + exog prep
#       Tier B — Multiplier-based scenario shocks
# ==================================================

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


# ── FRED series catalogue ─────────────────────────────────────────────────────
FRED_CATALOGUE = {
    # Interest Rates
    "FEDFUNDS":  {"label": "Federal Funds Rate",        "category": "Interest Rates", "unit": "%"},
    "GS10":      {"label": "10-Year Treasury Yield",    "category": "Interest Rates", "unit": "%"},
    "GS2":       {"label": "2-Year Treasury Yield",     "category": "Interest Rates", "unit": "%"},
    "MORTGAGE30US": {"label": "30-Year Mortgage Rate",  "category": "Interest Rates", "unit": "%"},
    "BAMLH0A0HYM2": {"label": "High Yield Credit Spread","category": "Credit",       "unit": "%"},

    # Labor Market
    "UNRATE":    {"label": "Unemployment Rate",         "category": "Labor",          "unit": "%"},
    "PAYEMS":    {"label": "Nonfarm Payrolls",          "category": "Labor",          "unit": "K"},
    "ICSA":      {"label": "Initial Jobless Claims",    "category": "Labor",          "unit": "K"},

    # Inflation & Prices
    "CPIAUCSL":  {"label": "CPI (All Items)",           "category": "Inflation",      "unit": "Index"},
    "CPILFESL":  {"label": "Core CPI (ex Food & Energy)","category": "Inflation",    "unit": "Index"},
    "PCEPI":     {"label": "PCE Price Index",           "category": "Inflation",      "unit": "Index"},
    "PPIFIS":    {"label": "PPI (Final Demand)",        "category": "Inflation",      "unit": "Index"},

    # Growth & Activity
    "GDP":       {"label": "Real GDP",                  "category": "Growth",         "unit": "B$"},
    "INDPRO":    {"label": "Industrial Production",     "category": "Growth",         "unit": "Index"},
    "RETAILSMNSA": {"label": "Retail Sales",            "category": "Growth",         "unit": "M$"},
    "UMCSENT":   {"label": "Consumer Sentiment",        "category": "Growth",         "unit": "Index"},

    # Credit & Financial
    "DRCCLACBS": {"label": "Credit Card Delinquency Rate","category": "Credit",       "unit": "%"},
    "TOTALSL":   {"label": "Total Consumer Credit",    "category": "Credit",          "unit": "B$"},
    "DPSACBW027SBOG": {"label": "Bank Deposits",       "category": "Credit",          "unit": "B$"},

    # Housing
    "HOUST":     {"label": "Housing Starts",           "category": "Housing",         "unit": "K"},
    "CSUSHPINSA": {"label": "Case-Shiller Home Price",  "category": "Housing",        "unit": "Index"},

    # Global / FX
    "DTWEXBGS":  {"label": "USD Trade-Weighted Index", "category": "FX",              "unit": "Index"},
    "DCOILWTICO": {"label": "WTI Crude Oil Price",      "category": "Commodities",    "unit": "$/bbl"},
}

# Tier B — preset multipliers (% shock on forecast per 1 unit change in variable)
TIER_B_PRESETS = {
    "Federal Funds Rate":    {"default_shock": 0.0, "multiplier": -2.5,
                               "description": "1% rate hike → ~2.5% demand headwind"},
    "Unemployment Rate":     {"default_shock": 0.0, "multiplier": -3.0,
                               "description": "1pp unemployment rise → ~3% activity drag"},
    "CPI Inflation":         {"default_shock": 0.0, "multiplier": -1.5,
                               "description": "1% CPI rise → ~1.5% real spending drag"},
    "Credit Spread (HY)":    {"default_shock": 0.0, "multiplier": -1.8,
                               "description": "100bp spread widening → ~1.8% tightening effect"},
    "GDP Growth":            {"default_shock": 0.0, "multiplier": +2.0,
                               "description": "1pp GDP growth → ~2% demand uplift"},
    "Consumer Sentiment":    {"default_shock": 0.0, "multiplier": +1.2,
                               "description": "10-point sentiment rise → ~1.2% spending lift"},
    "Oil Price (WTI)":       {"default_shock": 0.0, "multiplier": -0.8,
                               "description": "$10/bbl oil rise → ~0.8% cost drag"},
    "USD Strength Index":    {"default_shock": 0.0, "multiplier": -1.0,
                               "description": "5% USD appreciation → ~1% export headwind"},
}


def fetch_fred_series(
    series_id: str,
    fred_api_key: str,
    start_date: str = "2000-01-01",
) -> Optional[pd.Series]:
    """
    Fetch a single FRED series via API.
    Returns a pd.Series indexed by date, or None on failure.
    """
    try:
        import urllib.request, json
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={fred_api_key}"
            f"&file_type=json"
            f"&observation_start={start_date}"
            f"&frequency=m"  # monthly
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())

        obs = data.get("observations", [])
        if not obs:
            return None

        dates  = pd.to_datetime([o["date"] for o in obs])
        values = pd.to_numeric(
            [o["value"] for o in obs],
            errors="coerce"
        )
        s = pd.Series(values.values, index=dates, name=series_id)
        return s.dropna()

    except Exception:
        return None


def build_exog_df(
    committed_df: pd.DataFrame,
    selected_series: list,
    fred_api_key: str,
) -> tuple[pd.DataFrame | None, list]:
    """
    Fetch all selected FRED series and align to committed_df date range.
    Returns (exog_df, errors_list).
    exog_df has date index + one column per series.
    """
    dates    = pd.to_datetime(committed_df["date"])
    start    = dates.min().strftime("%Y-%m-%d")
    errors   = []
    series_dict = {}

    for sid in selected_series:
        s = fetch_fred_series(sid, fred_api_key, start_date=start)
        if s is None:
            errors.append(f"Could not fetch {sid}")
            continue
        # Resample to month-start to align with committed data
        s = s.resample("MS").last().ffill()
        series_dict[sid] = s

    if not series_dict:
        return None, errors

    exog = pd.DataFrame(series_dict)

    # Align to committed dates
    target_idx = pd.DatetimeIndex(dates.values)
    exog = exog.reindex(target_idx, method="ffill")

    # Normalise each column: (x - mean) / std
    for col in exog.columns:
        mu, sigma = exog[col].mean(), exog[col].std()
        if sigma > 0:
            exog[col] = (exog[col] - mu) / sigma

    return exog, errors


def apply_tier_b_multipliers(
    baseline: pd.Series,
    multipliers: dict,
) -> pd.Series:
    """
    Apply Tier B multiplier shocks to baseline forecast.

    multipliers: {var_name: assumed_shock_value}
    Each shock is scaled by the preset multiplier to get a % impact.
    Total impact is summed and applied multiplicatively.
    """
    total_pct_impact = 0.0
    for var_name, shock_val in multipliers.items():
        if var_name not in TIER_B_PRESETS or shock_val == 0:
            continue
        mult = TIER_B_PRESETS[var_name]["multiplier"]
        total_pct_impact += (shock_val * mult / 100.0)

    if total_pct_impact == 0:
        return baseline.copy()

    return baseline * (1.0 + total_pct_impact)


def get_categories() -> dict:
    """Return FRED_CATALOGUE grouped by category."""
    cats = {}
    for sid, meta in FRED_CATALOGUE.items():
        cat = meta["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append((sid, meta["label"], meta["unit"]))
    return cats
